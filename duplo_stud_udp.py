#!/usr/bin/env python3
"""
duplo_stud_udp.py — 듀플로 브릭 스터드 중심점 검출 (RealSense D455)
====================================================================
파이프라인
    RealSense(1280x800 RGB + aligned depth)
      → GroundingDINO   : 브릭 후보 박스 (스터드가 아니라 '브릭'을 찾음)
      → SAM2            : 브릭 인스턴스 마스크
      → [A] 윗면 분리   : depth 기반 RANSAC 평면 fitting
      → [B] minAreaRect : 윗면 사각형 → 변 길이(px) → mm
      → [C] 격자 스냅   : 16mm 피치로 nx, ny 정수 확정  (노이즈 제거 핵심)
      → [D] solvePnP    : 알려진 치수로 6-DoF pose (depth보다 정확)
      → [E] 정사영 rectify + Hough ring → 실측 스터드로 in-plane 3-DOF 보정
      → base 좌표 변환 (hand-eye T_base_cam) → UDP 송신

중요
    · SAM2는 '브릭 마스크'까지만 담당합니다. 격자/스터드는 전부 기하 계산입니다.
    · X_OFFSET/Y_OFFSET 하드코딩을 폐기하고 4x4 변환행렬을 씁니다.
      handeye_T_base_cam.npy 가 없으면 단위행렬(=카메라 좌표) 로 동작하며 경고합니다.
      이 캘리브레이션 오차가 전체 오차를 지배하므로 반드시 수행하십시오.

실행
    conda activate env_isaaclab
    python3 duplo_stud_udp.py

키
    q : 종료          r : 재검출 강제      h : Hough 보정 on/off
    m : 마스크 표시   d : 디버그(rectify 창)  p : 프롬프트 입력 안내
"""

import sys
import json
import time
import socket
import threading

import numpy as np
import cv2
import torch
import pyrealsense2 as rs

# ════════════════════════════════════════════════════════════════════
# 1. CONFIG  — 실측값으로 반드시 교체하십시오
# ════════════════════════════════════════════════════════════════════

# ── 듀플로 물리 치수 (mm) ──────────────────────────────────────────
# 캘리퍼로 실측 후 교체할 것. 이 값들이 전체 파이프라인의 기준입니다.
STUD_PITCH_MM = 16.0     # 스터드 피치 (확실)
STUD_DIA_MM   = 9.35     # 스터드 외경 (중공 링)
STUD_H_MM     = 4.5      # 스터드 돌출 높이  ← 실측 권장
BRICK_H_MM    = 19.2     # 브릭 높이
FOOTPRINT_GAP = 0.2      # 풋프린트 = n*PITCH - GAP  (2x4 → 31.8 x 63.8)

# ── 검출 ──────────────────────────────────────────────────────────
# 'lego stud' 는 GroundingDINO 학습분포 밖이라 실패합니다. 브릭 단위로 찾으세요.
TEXT_PROMPT    = "toy block."
BOX_THRESHOLD  = 0.30
TEXT_THRESHOLD = 0.25
MAX_BRICKS     = 8

# ── 카메라 ────────────────────────────────────────────────────────
COLOR_W, COLOR_H, COLOR_FPS = 1280, 800, 30   # 640x480은 정보를 절반 이상 버림
DEPTH_W, DEPTH_H, DEPTH_FPS = 848, 480, 30    # D455 native

# ── 윗면 분리 ─────────────────────────────────────────────────────
NEARER_PLANE_MM  = 8.0    # 이보다 앞쪽에 큰 면이 있으면 그쪽을 윗면으로 재판정
                          # (스터드 4.5mm 는 통과, 테이블 위 브릭 19.2mm 는 포착)
RANSAC_THRESH_MM = 2.0
RANSAC_ITERS     = 200
USE_HOLE_FILLING = False  # hole filling 은 없는 depth 를 '지어냅니다'.
                          # 평면 fitting 을 오염시키므로 정밀 작업에선 off 권장.
MIN_TOP_PIXELS   = 400

# ── 격자 스냅 검증 ────────────────────────────────────────────────
GRID_RESIDUAL_MAX = 0.28   # |a/16 - round(a/16)| 이보다 크면 reject
MAX_STUD_N        = 8      # 한 변 최대 스터드 수

# ── Hough 역검증 (rectified 정사영 이미지에서 수행) ────────────────
USE_HOUGH_REFINE = True
RECT_PX_PER_MM   = 4.0     # 정사영 해상도 (스터드 반경 ≈ 18.7px 가 됨)
RECT_MARGIN_MM   = 6.0
HOUGH_MATCH_TOL_MM = 5.0   # 이 안에 들어와야 nominal-measured 매칭
HOUGH_MIN_MATCH_RATIO = 0.5

# ── 출력 ──────────────────────────────────────────────────────────
UDP_IP, UDP_PORT = "127.0.0.1", 5005
UDP_FORMAT = "json"        # "json" | "legacy"  (legacy = "x,y" 문자열)
HANDEYE_PATH = "handeye_T_base_cam.npy"   # 4x4, base <- cam

# ── 성능 ──────────────────────────────────────────────────────────
DETECT_EVERY = 4           # N 프레임마다 무거운 검출 (그 사이는 캐시 표시)

SAM2_BASE_PATH = "/home/kimjimin/sam2"
SAM2_CKPT      = SAM2_BASE_PATH + "/checkpoints/sam2.1_hiera_tiny.pt"
SAM2_CFG       = "configs/sam2.1/sam2.1_hiera_t.yaml"
GDINO_ID       = "IDEA-Research/grounding-dino-tiny"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════════════
# 2. 기하 유틸
# ════════════════════════════════════════════════════════════════════

def intr_to_K(intr):
    """RealSense intrinsics → OpenCV 카메라 행렬."""
    return np.array([[intr.fx, 0.0, intr.ppx],
                     [0.0, intr.fy, intr.ppy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def deproject_mask(mask, depth_m, K):
    """마스크 픽셀 → 카메라 좌표계 3D 점(mm) + 픽셀 인덱스."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None, None
    z = depth_m[ys, xs]
    ok = z > 0
    if ok.sum() == 0:
        return None, None, None
    xs, ys, z = xs[ok], ys[ok], z[ok] * 1000.0          # mm
    X = (xs - K[0, 2]) * z / K[0, 0]
    Y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([X, Y, z], axis=1), xs, ys


def ransac_plane(pts, thresh, iters, rng):
    """3점 RANSAC 평면 fitting → (normal, d, inlier_mask). 평면: n·p + d = 0"""
    n_pts = len(pts)
    if n_pts < 3:
        return None, None, None
    best_inl, best_n, best_d = None, None, None
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-6:
            continue
        nrm = nrm / norm
        d = -float(nrm @ p0)
        dist = np.abs(pts @ nrm + d)
        inl = dist < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_n, best_d = inl, nrm, d
    if best_inl is None or best_inl.sum() < 3:
        return None, None, None
    # inlier 로 최소자승 재추정 (SVD)
    P = pts[best_inl]
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    nrm = Vt[2] / (np.linalg.norm(Vt[2]) + 1e-12)
    d = -float(nrm @ c)
    inl = np.abs(pts @ nrm + d) < thresh
    return nrm, d, inl


def rigid_2d(src, dst):
    """2D 대응점 → 회전+평행이동 (스케일 없음, Kabsch). 반환 (R2, t2)."""
    if len(src) < 2:
        return np.eye(2), np.zeros(2)
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    H = (src - cs).T @ (dst - cd)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return R, cd - R @ cs


# ════════════════════════════════════════════════════════════════════
# 3. 브릭 pose 추정
# ════════════════════════════════════════════════════════════════════

class BrickResult:
    """브릭 하나의 추정 결과 컨테이너."""

    def __init__(self):
        self.ok = False
        self.reason = ""
        self.nx = self.ny = 0
        self.rvec = self.tvec = None
        self.R = None
        self.normal = None          # 카메라를 향하는 윗면 법선 (단위)
        self.studs_local = None     # (N,2) 로컬 mm  (보정 반영)
        self.studs_cam = None       # (N,3) 카메라 mm — 스터드 상면 중심
        self.studs_px = None        # (N,2) 이미지 픽셀
        self.corners_px = None      # (4,2)
        self.top_mask = None
        self.n_matched = 0
        self.z_mm = 0.0
        self.rect_debug = None


def extract_top_face(mask, depth_m, K, rng):
    """마스크 안에서 '윗면' 픽셀만 남긴다.

    옆면은 depth 가 더 멀고 법선이 다르므로 near-cluster + RANSAC 평면으로 분리.
    듀플로 브릭 높이 19.2mm 는 D455 depth 노이즈(~0.6mm @0.5m)의 30배라 분리 가능.
    """
    pts, xs, ys = deproject_mask(mask, depth_m, K)
    if pts is None or len(pts) < MIN_TOP_PIXELS:
        return None, None, None, None

    # 고정 깊이 밴드를 쓰면 안 됩니다. 2x4 브릭(64mm)이 15° 기울면 윗면
    # 자체의 깊이 폭이 16.6mm 라서 밴드가 윗면을 잘라냅니다.
    # 대신 전체에 RANSAC → 더 앞쪽에 큰 면이 남아 있으면 그쪽으로 재수행.
    sub = np.arange(len(pts))
    nrm = d = None
    for _ in range(3):
        nrm, d, inl = ransac_plane(pts[sub], RANSAC_THRESH_MM,
                                   RANSAC_ITERS, rng)
        if nrm is None:
            return None, None, None, None
        if nrm[2] > 0:                      # 법선을 카메라 쪽으로 (n_z < 0)
            nrm, d = -nrm, -d
        sd = pts @ nrm + d                  # > 0 이면 평면보다 카메라에 가까움
        nearer = sd > NEARER_PLANE_MM       # 스터드(4.5mm)는 제외, 테이블 위 브릭(19.2mm)은 포착
        if nearer.sum() > MIN_TOP_PIXELS and \
           nearer.sum() > 0.45 * len(pts):
            sub = np.nonzero(nearer)[0]     # 앞쪽 면이 진짜 윗면
            continue
        sel_local = np.abs(pts @ nrm + d) < RANSAC_THRESH_MM
        break

    sel = sel_local
    if sel.sum() < MIN_TOP_PIXELS:
        return None, None, None, None

    top = np.zeros(mask.shape, dtype=np.uint8)
    top[ys[sel], xs[sel]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    top = cv2.morphologyEx(top, cv2.MORPH_CLOSE, k, iterations=2)
    top = cv2.morphologyEx(top, cv2.MORPH_OPEN, k, iterations=1)
    return top, nrm, float(d), float(np.median(pts[sel, 2]))


def snap_grid(len_mm):
    """변 길이(mm) → 스터드 개수 정수. 실패 시 (None, residual)."""
    raw = (len_mm + FOOTPRINT_GAP) / STUD_PITCH_MM
    n = int(round(raw))
    if n < 1 or n > MAX_STUD_N:
        return None, abs(raw - n)
    return n, abs(raw - n)


def solve_brick_pose(top_mask, K, n_plane, d_plane):
    """윗면 마스크 + 평면방정식 → 6-DoF pose + 격자 크기.

    ※ 설계 근거 (중요)
      4개 코너만으로 solvePnP 를 돌리면 안 됩니다. 0.4m 에서 32x64mm 사각형은
      원근 효과가 거의 없어 out-of-plane 회전이 ill-conditioned 이고,
      재투영 1~2px 오차에도 3D 오차가 10mm 이상 튑니다 (planar pose ambiguity).

      대신 역할을 분리합니다.
        · 기울기(2 DOF) ← depth RANSAC 평면. 수천 점 평균이라 훨씬 안정적.
        · 면내(3 DOF)   ← 이미지 사각형 코너를 그 평면에 ray-plane 교차.
      스터드는 평면 위에 있으므로 기울기 오차의 레버암은 브릭 크기가 아니라
      스터드 높이(4.5mm) 뿐입니다. 기울기 1° 오차 → 횡방향 0.08mm.

    반환 (R, t, na, nb, box_px)  — R = [f1 f2 e3], t = 윗면 중심(mm, cam)
    """
    cnts, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, "no contour"
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < MIN_TOP_PIXELS:
        return None, "top too small"

    # ── 윤곽선 전 픽셀 → 시선(ray) → 윗면 평면과 교차 → 3D(mm) ──
    # 기울어진 직사각형은 이미지에서 사다리꼴로 투영되므로 이미지 공간
    # minAreaRect 는 크기를 부풀립니다. 평면에 먼저 투영해 원근을 제거합니다.
    pts_px = cnt.reshape(-1, 2).astype(np.float64)
    Kinv = np.linalg.inv(K)
    rays = (Kinv @ np.column_stack([pts_px, np.ones(len(pts_px))]).T).T
    den = rays @ n_plane
    good = np.abs(den) > 1e-9
    s = np.zeros(len(rays))
    s[good] = -d_plane / den[good]
    good &= s > 0
    if good.sum() < 8:
        return None, "ray/plane fail"
    P = rays[good] * s[good, None]                      # (N,3) mm, 평면 위

    # ── 평면 위 미터법 2D 좌표계 (임의 기저) ──
    e3 = n_plane / np.linalg.norm(n_plane)              # 카메라 방향
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(e3 @ tmp) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    g1 = np.cross(e3, tmp); g1 /= np.linalg.norm(g1)
    g2 = np.cross(e3, g1);  g2 /= np.linalg.norm(g2)

    origin = P.mean(axis=0)
    D = P - origin
    Q = np.stack([D @ g1, D @ g2], axis=1).astype(np.float32)   # mm

    rect = cv2.minAreaRect(Q)
    (qc1, qc2), (qw, qh), _ = rect
    if qw < 1e-3 or qh < 1e-3:
        return None, "degenerate rect"

    # 채움률 — 원근이 제거된 공간에서 계산하므로 이제 유효한 판정
    fill = abs(cv2.contourArea(Q)) / (qw * qh)
    if fill < 0.80:
        return None, f"low fill {fill:.2f}"

    # 각도 규약에 의존하지 않도록 boxPoints 에서 직접 방향을 뽑음
    q = cv2.boxPoints(rect).astype(np.float64)
    du, dv = q[1] - q[0], q[2] - q[1]
    A_mm, B_mm = float(np.linalg.norm(du)), float(np.linalg.norm(dv))

    na, ra = snap_grid(A_mm)
    nb, rb = snap_grid(B_mm)
    if na is None or nb is None:
        return None, "grid out of range"
    if max(ra, rb) > GRID_RESIDUAL_MAX:
        return None, f"grid residual {max(ra, rb):.2f}"

    f1 = (du[0] * g1 + du[1] * g2); f1 /= np.linalg.norm(f1)
    f2 = (dv[0] * g1 + dv[1] * g2); f2 /= np.linalg.norm(f2)
    f2 = f2 - (f2 @ f1) * f1; f2 /= (np.linalg.norm(f2) + 1e-12)

    R = np.column_stack([f1, f2, e3])
    if np.linalg.det(R) < 0:            # 좌수계면 f2 부호 반전 (격자는 대칭)
        f2 = -f2
        R = np.column_stack([f1, f2, e3])

    t = origin + qc1 * g1 + qc2 * g2     # 윗면 중심 (카메라 좌표, mm)

    # 시각화용 코너 픽셀 (스냅된 정규 치수로 재생성)
    A_s = na * STUD_PITCH_MM - FOOTPRINT_GAP
    B_s = nb * STUD_PITCH_MM - FOOTPRINT_GAP
    corn3d = np.array([t - f1 * A_s / 2 - f2 * B_s / 2,
                       t + f1 * A_s / 2 - f2 * B_s / 2,
                       t + f1 * A_s / 2 + f2 * B_s / 2,
                       t - f1 * A_s / 2 + f2 * B_s / 2])
    cp = (K @ corn3d.T).T
    box = cp[:, :2] / cp[:, 2:3]
    return (R, t, na, nb, box), None


def nominal_studs(na, nb):
    """로컬 평면 좌표계의 스터드 중심 격자 (na x nb, mm)."""
    u = (np.arange(na) - (na - 1) / 2.0) * STUD_PITCH_MM
    v = (np.arange(nb) - (nb - 1) / 2.0) * STUD_PITCH_MM
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.stack([uu.ravel(), vv.ravel()], axis=1)


# ════════════════════════════════════════════════════════════════════
# 4. Hough 역검증 / 보정
# ════════════════════════════════════════════════════════════════════

_HAS_HOUGH_ALT = hasattr(cv2, "HOUGH_GRADIENT_ALT")


def rectify_top(image_bgr, K, dist, R, t, A, B):
    """윗면을 정사영(mm 스케일) 이미지로 warp.

    평면 z=0 이므로 H = K [f1 f2 t]. 여기에 mm→rect 픽셀 스케일을 결합.
    정사영에서 스터드는 완전한 원 + 정확한 격자가 되어 Hough 신뢰도가 크게 오름.
    """
    H_img_local = K @ np.column_stack([R[:, 0], R[:, 1], np.asarray(t).ravel()])

    s = RECT_PX_PER_MM
    w = int((A + 2 * RECT_MARGIN_MM) * s)
    h = int((B + 2 * RECT_MARGIN_MM) * s)
    S = np.array([[s, 0.0, w / 2.0],
                  [0.0, s, h / 2.0],
                  [0.0, 0.0, 1.0]])            # local mm → rect px
    H_img_rect = H_img_local @ np.linalg.inv(S)

    # 렌즈 왜곡 보정 후 warp (D455 RGB 는 왜곡이 작아 dist=0 이면 no-op)
    src = image_bgr if dist is None or not np.any(dist) else \
        cv2.undistort(image_bgr, K, dist)
    rect = cv2.warpPerspective(src, np.linalg.inv(H_img_rect), (w, h),
                               flags=cv2.INTER_LINEAR)
    return rect, S


def detect_studs_rectified(rect_bgr):
    """정사영 이미지에서 스터드 링 검출 → 로컬 mm 좌표."""
    s = RECT_PX_PER_MM
    r_px = STUD_DIA_MM * 0.5 * s
    pitch_px = STUD_PITCH_MM * s

    gray = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    if _HAS_HOUGH_ALT:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT_ALT, dp=1.5,
            minDist=pitch_px * 0.6, param1=300, param2=0.72,
            minRadius=int(r_px * 0.55), maxRadius=int(r_px * 1.6))
    else:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1,
            minDist=pitch_px * 0.6, param1=110, param2=20,
            minRadius=int(r_px * 0.55), maxRadius=int(r_px * 1.6))

    if circles is None:
        return np.empty((0, 2)), gray

    c = circles[0]
    h, w = gray.shape
    meas = np.stack([(c[:, 0] - w / 2.0) / s,
                     (c[:, 1] - h / 2.0) / s], axis=1)   # local mm
    return meas, gray


def refine_by_studs(nominal, measured):
    """nominal→measured 최근접 매칭 후 2D 강체변환 fitting.

    Hough 가 절반만 찾아도 격자가 기지이므로 나머지를 보간하고,
    격자 밖 오검출은 매칭 tolerance 로 자동 제거됩니다.
    """
    if len(measured) == 0:
        return nominal, 0
    pairs_s, pairs_d = [], []
    used = set()
    for p in nominal:
        d = np.linalg.norm(measured - p, axis=1)
        j = int(np.argmin(d))
        if d[j] < HOUGH_MATCH_TOL_MM and j not in used:
            used.add(j)
            pairs_s.append(p)
            pairs_d.append(measured[j])
    n = len(pairs_s)
    if n < max(2, int(len(nominal) * HOUGH_MIN_MATCH_RATIO)):
        return nominal, n
    R2, t2 = rigid_2d(np.array(pairs_s), np.array(pairs_d))
    return nominal @ R2.T + t2, n


# ════════════════════════════════════════════════════════════════════
# 5. 브릭 1개 전체 처리
# ════════════════════════════════════════════════════════════════════

def process_brick(mask, image_bgr, depth_m, K, dist, rng, use_hough):
    res = BrickResult()

    top, nrm, d_pl, z_mm = extract_top_face(mask, depth_m, K, rng)
    if top is None:
        res.reason = "top face fail"
        return res
    res.top_mask, res.z_mm = top, z_mm

    out, err = solve_brick_pose(top, K, nrm, d_pl)
    if out is None:
        res.reason = err
        return res
    R, t, na, nb, box = out

    A = na * STUD_PITCH_MM - FOOTPRINT_GAP
    B = nb * STUD_PITCH_MM - FOOTPRINT_GAP
    local = nominal_studs(na, nb)

    if use_hough:
        rect, _ = rectify_top(image_bgr, K, dist, R, t, A, B)
        meas, gray = detect_studs_rectified(rect)
        local, n_match = refine_by_studs(local, meas)
        res.n_matched = n_match
        dbg = rect.copy()
        for m in meas:
            p = (int(m[0] * RECT_PX_PER_MM + rect.shape[1] / 2),
                 int(m[1] * RECT_PX_PER_MM + rect.shape[0] / 2))
            cv2.circle(dbg, p, int(STUD_DIA_MM * 0.5 * RECT_PX_PER_MM),
                       (0, 255, 255), 1)
        for p2 in local:
            p = (int(p2[0] * RECT_PX_PER_MM + rect.shape[1] / 2),
                 int(p2[1] * RECT_PX_PER_MM + rect.shape[0] / 2))
            cv2.drawMarker(dbg, p, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        res.rect_debug = dbg

    # e3 는 이미 카메라를 향하도록 정규화되어 있음
    n_cam = R[:, 2]

    # 로컬(u,v,0) → 카메라, 그 뒤 법선 방향으로 스터드 높이만큼 올림
    plate = (R[:, :2] @ local.T).T + t                 # 플레이트면 중심
    studs_cam = plate + n_cam * STUD_H_MM              # 스터드 상면 중심

    px, _ = cv2.projectPoints(studs_cam.astype(np.float64),
                              np.zeros(3), np.zeros(3), K, dist)

    res.ok = True
    res.nx, res.ny = na, nb
    res.rvec, res.tvec = cv2.Rodrigues(R)[0], t.reshape(3, 1)
    res.R, res.normal = R, n_cam
    res.studs_local = local
    res.studs_cam = studs_cam
    res.studs_px = px.reshape(-1, 2)
    res.corners_px = box
    return res


# ════════════════════════════════════════════════════════════════════
# 6. 검출기 (GroundingDINO + SAM2)
# ════════════════════════════════════════════════════════════════════

class Detector:
    def __init__(self):
        from transformers import (AutoProcessor,
                                  AutoModelForZeroShotObjectDetection)
        print("🔄 GroundingDINO 로딩...")
        self.gp = AutoProcessor.from_pretrained(GDINO_ID)
        self.gm = AutoModelForZeroShotObjectDetection \
            .from_pretrained(GDINO_ID).to(DEVICE)
        print("✅ GroundingDINO")

        sys.path.append(SAM2_BASE_PATH)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("🔄 SAM2 로딩...")
        self.sam = SAM2ImagePredictor(
            build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE))
        print("✅ SAM2")

    def boxes(self, image_rgb, prompt):
        inp = self.gp(images=image_rgb, text=prompt,
                      return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = self.gm(**inp)
        r = self.gp.post_process_grounded_object_detection(
            out, inp.input_ids,
            threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[image_rgb.shape[:2]])[0]
        b, s = r["boxes"], r["scores"]
        if len(b) == 0:
            return np.empty((0, 4), dtype=np.float32)
        order = torch.argsort(s, descending=True)[:MAX_BRICKS]
        return b[order].cpu().numpy().astype(np.float32)

    def masks(self, image_rgb, boxes):
        """박스 전체를 한 번에 처리 — set_image 는 프레임당 1회만."""
        if len(boxes) == 0:
            return []
        import contextlib
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if DEVICE == "cuda" else contextlib.nullcontext())
        with torch.inference_mode(), amp:
            self.sam.set_image(image_rgb)     # 프레임당 1회 (인코더 재실행 방지)
            m, _, _ = self.sam.predict(box=boxes, multimask_output=False)
        m = np.asarray(m)
        if m.ndim == 4:                              # (N,1,H,W)
            m = m[:, 0]
        elif m.ndim == 3 and len(boxes) == 1 and m.shape[0] != 1:
            m = m[:1]                                # (num_masks,H,W) → 1개
        elif m.ndim == 2:                            # (H,W)
            m = m[None]
        return [mm.astype(bool) for mm in m]


# ════════════════════════════════════════════════════════════════════
# 7. 좌표 변환 / 송신
# ════════════════════════════════════════════════════════════════════

def load_handeye(path):
    try:
        T = np.load(path)
        assert T.shape == (4, 4)
        print(f"✅ hand-eye 로드: {path}")
        return T, True
    except Exception:
        print("⚠️  hand-eye 캘리브레이션 파일 없음 → 카메라 좌표계로 출력합니다.")
        print("    cv2.calibrateHandEye() 결과를 4x4 로 저장해 두십시오.")
        print(f"    경로: {path}")
        print("    1° 오차 = 0.5m 에서 8.7mm — 스터드 하나가 통째로 어긋납니다.")
        return np.eye(4), False


def cam_to_base(pts_mm, T):
    """(N,3) mm 카메라 → (N,3) m base."""
    p = np.asarray(pts_mm, dtype=np.float64) / 1000.0
    return (T[:3, :3] @ p.T).T + T[:3, 3]


def send(sock, results, T):
    if UDP_FORMAT == "legacy":
        for r in results:
            if r.ok and len(r.studs_cam):
                b = cam_to_base(r.studs_cam[:1], T)[0]
                sock.sendto(f"{b[0]:.4f},{b[1]:.4f}".encode(),
                            (UDP_IP, UDP_PORT))
                return
        return

    payload = {"t": time.time(), "frame": "base", "unit": "m", "bricks": []}
    for r in results:
        if not r.ok:
            continue
        b = cam_to_base(r.studs_cam, T)
        payload["bricks"].append({
            "nx": r.nx, "ny": r.ny,
            "n_hough_matched": r.n_matched,
            "studs": [[round(v, 5) for v in p] for p in b.tolist()],
        })
    sock.sendto(json.dumps(payload).encode(), (UDP_IP, UDP_PORT))


# ════════════════════════════════════════════════════════════════════
# 8. 프롬프트 입력 스레드
# ════════════════════════════════════════════════════════════════════

_prompt_lock = threading.Lock()
_prompt = TEXT_PROMPT


def get_prompt():
    with _prompt_lock:
        return _prompt


def input_thread():
    global _prompt
    while True:
        try:
            s = input().strip()
        except EOFError:
            break
        if not s:
            continue
        if not s.endswith("."):
            s += "."
        with _prompt_lock:
            _prompt = s.lower()
        print(f"🎯 대상 변경 → '{_prompt}'")


# ════════════════════════════════════════════════════════════════════
# 9. 시각화
# ════════════════════════════════════════════════════════════════════

def draw(vis, results, show_mask, use_hough, fps):
    for r in results:
        if not r.ok:
            continue
        if show_mask and r.top_mask is not None:
            ov = vis.copy()
            ov[r.top_mask > 0] = (0, 200, 0)
            vis[:] = cv2.addWeighted(vis, 0.75, ov, 0.25, 0)

        cv2.polylines(vis, [r.corners_px.astype(np.int32)], True,
                      (255, 160, 0), 2)

        for i, p in enumerate(r.studs_px):
            x, y = int(round(p[0])), int(round(p[1]))
            if not (0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]):
                continue
            cv2.circle(vis, (x, y), 11, (0, 255, 255), 2)
            cv2.drawMarker(vis, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 9, 2)
            cv2.putText(vis, str(i), (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        c = r.corners_px.mean(axis=0).astype(int)
        cv2.putText(vis, f"{r.nx}x{r.ny}  z={r.z_mm:.0f}mm  H:{r.n_matched}",
                    (c[0] - 70, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)

    n_ok = sum(1 for r in results if r.ok)
    n_st = sum(len(r.studs_cam) for r in results if r.ok)
    bar = (f"{fps:4.1f}fps | bricks {n_ok} | studs {n_st} | "
           f"hough {'ON' if use_hough else 'OFF'} | '{get_prompt()}'")
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, bar, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 128), 1)

    fails = [r.reason for r in results if not r.ok and r.reason]
    for i, f in enumerate(fails[:3]):
        cv2.putText(vis, f"reject: {f}", (8, 50 + 20 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)


# ════════════════════════════════════════════════════════════════════
# 10. main
# ════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(0)
    T_base_cam, _ = load_handeye(HANDEYE_PATH)
    det = Detector()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_W, COLOR_H,
                      rs.format.bgr8, COLOR_FPS)
    cfg.enable_stream(rs.stream.depth, DEPTH_W, DEPTH_H,
                      rs.format.z16, DEPTH_FPS)
    profile = pipeline.start(cfg)

    # High Accuracy 프리셋 — 근접 정밀 작업에 유리
    try:
        dev = profile.get_device().first_depth_sensor()
        if dev.supports(rs.option.visual_preset):
            dev.set_option(rs.option.visual_preset, 3)
    except Exception:
        pass

    align = rs.align(rs.stream.color)
    spatial, temporal = rs.spatial_filter(), rs.temporal_filter()
    hole = rs.hole_filling_filter() if USE_HOLE_FILLING else None
    intr = profile.get_stream(rs.stream.color) \
        .as_video_stream_profile().get_intrinsics()
    K = intr_to_K(intr)
    dist = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)[:5]
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    print(f"🎥 {COLOR_W}x{COLOR_H}  fx={intr.fx:.1f}  "
          f"mm/px@0.4m={0.4*1000/intr.fx:.2f}")
    print(f"📡 UDP → {UDP_IP}:{UDP_PORT} ({UDP_FORMAT})")
    print("⌨️  프롬프트 입력 후 Enter (예: yellow toy block)")
    threading.Thread(target=input_thread, daemon=True).start()

    results, use_hough, show_mask, show_dbg = [], USE_HOUGH_REFINE, True, False
    frame_i, t_last, fps = 0, time.time(), 0.0

    try:
        while True:
            fs = align.process(pipeline.wait_for_frames())
            cf, df = fs.get_color_frame(), fs.get_depth_frame()
            if not cf or not df:
                continue

            df = temporal.process(spatial.process(df))
            if hole is not None:
                df = hole.process(df)
            image_bgr = np.asanyarray(cf.get_data())
            depth_m = np.asanyarray(df.get_data()).astype(np.float32) \
                * depth_scale

            if frame_i % DETECT_EVERY == 0:
                rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                boxes = det.boxes(rgb, get_prompt())
                masks = det.masks(rgb, boxes)
                results = [process_brick(m, image_bgr, depth_m, K, dist,
                                         rng, use_hough) for m in masks]
                if any(r.ok for r in results):
                    send(sock, results, T_base_cam)
                now = time.time()
                fps = DETECT_EVERY / max(now - t_last, 1e-6)
                t_last = now

            vis = image_bgr.copy()
            draw(vis, results, show_mask, use_hough, fps)
            cv2.imshow("Duplo Stud Detector", vis)

            if show_dbg:
                for r in results:
                    if r.ok and r.rect_debug is not None:
                        cv2.imshow("rectified top (debug)", r.rect_debug)
                        break

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('h'):
                use_hough = not use_hough
            elif k == ord('m'):
                show_mask = not show_mask
            elif k == ord('d'):
                show_dbg = not show_dbg
                if not show_dbg:
                    cv2.destroyWindow("rectified top (debug)")
            elif k == ord('r'):
                frame_i = -1
            elif k == ord('p'):
                print("⌨️  터미널에 프롬프트 입력 후 Enter")

            frame_i += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()

# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20 # 20
        self._std_weight_velocity = 1.0 / 50 # 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha

# class ExtendedKalmanFilter:
#     """
#     EKF with CTRV motion model for DeepSORT-like tracking.

#     State x = [px, py, v, psi, omega, a, h]
#       px,py   - bbox center
#       v       - speed magnitude
#       psi     - heading (rad)
#       omega   - turn rate (rad/frame)
#       a, h    - aspect ratio and height (kept linear)

#     Measurement z = [px, py, a, h]
#     """

#     def __init__(self, dt=1.0):
#         self.dt = dt
#         self.ndim = 7
#         self.zdim = 4

#         # noise scale factors (подбери под свои данные)
#         self._std_weight_pos = 1.0 / 20
#         self._std_weight_size = 1.0 / 20
#         self._std_weight_vel = 1.0 / 10
#         self._std_weight_turn = 1.0 / 30

#     # ---------- helpers ----------
#     def _f(self, x):
#         """Nonlinear motion model f(x)."""
#         px, py, v, psi, omega, a, h = x
#         dt = self.dt

#         if np.abs(omega) < 1e-6:  # почти прямолинейно
#             px_n = px + v * np.cos(psi) * dt
#             py_n = py + v * np.sin(psi) * dt
#         else:
#             px_n = px + (v / omega) * (np.sin(psi + omega * dt) - np.sin(psi))
#             py_n = py - (v / omega) * (np.cos(psi + omega * dt) - np.cos(psi))

#         v_n = v  # конст. скорость
#         psi_n = psi + omega * dt
#         omega_n = omega  # конст. turn rate
#         a_n = a
#         h_n = h
#         return np.array([px_n, py_n, v_n, psi_n, omega_n, a_n, h_n], dtype=float)

#     def _F_jacobian(self, x):
#         """Jacobian of f wrt x."""
#         px, py, v, psi, omega, a, h = x
#         dt = self.dt
#         F = np.eye(self.ndim)

#         sin_psi = np.sin(psi)
#         cos_psi = np.cos(psi)

#         if np.abs(omega) < 1e-6:
#             # dpx/dv, dpx/dpsi
#             F[0, 2] = cos_psi * dt
#             F[0, 3] = -v * sin_psi * dt
#             # dpy/dv, dpy/dpsi
#             F[1, 2] = sin_psi * dt
#             F[1, 3] = v * cos_psi * dt
#         else:
#             w = omega
#             psi_wdt = psi + w * dt
#             sin_psi_wdt = np.sin(psi_wdt)
#             cos_psi_wdt = np.cos(psi_wdt)

#             F[0, 2] = (sin_psi_wdt - sin_psi) / w
#             F[0, 3] = (v / w) * (cos_psi_wdt - cos_psi)
#             F[0, 4] = (v * w * cos_psi_wdt * dt - v * (sin_psi_wdt - sin_psi)) / (w**2)

#             F[1, 2] = -(cos_psi_wdt - cos_psi) / w
#             F[1, 3] = (v / w) * (sin_psi_wdt - sin_psi)
#             F[1, 4] = (v * w * sin_psi_wdt * dt - v * (cos_psi_wdt - cos_psi)) / (w**2)

#             F[3, 4] = dt  # psi += omega*dt

#         return F

#     def _h(self, x):
#         """Measurement model: linear extraction of px, py, a, h."""
#         return np.array([x[0], x[1], x[5], x[6]], dtype=float)

#     def _H_jacobian(self, x):
#         H = np.zeros((self.zdim, self.ndim))
#         H[0, 0] = 1.0  # px
#         H[1, 1] = 1.0  # py
#         H[2, 5] = 1.0  # a
#         H[3, 6] = 1.0  # h
#         return H

#     # ---------- API, как у старого KF ----------
#     def initiate(self, measurement):
#         """
#         measurement: [px, py, a, h]
#         """
#         px, py, a, h = measurement
#         v = 0.0
#         psi = 0.0
#         omega = 0.0
#         mean = np.array([px, py, v, psi, omega, a, h], dtype=float)

#         std = [
#             2 * self._std_weight_pos * h,  # px
#             2 * self._std_weight_pos * h,  # py
#             5 * self._std_weight_vel * h,  # v
#             np.deg2rad(10),                # psi
#             np.deg2rad(5),                 # omega
#             1e-2,                          # a
#             2 * self._std_weight_size * h  # h
#         ]
#         cov = np.diag(np.square(std))
#         return mean, cov

#     def predict(self, mean, covariance):
#         F = self._F_jacobian(mean)
#         mean_pred = self._f(mean)

#         std_pos = self._std_weight_pos * mean[6]  # h
#         std_vel = self._std_weight_vel * mean[6]
#         std_turn = self._std_weight_turn
#         std_size = self._std_weight_size * mean[6]

#         Q = np.diag([
#             std_pos**2,       # px
#             std_pos**2,       # py
#             std_vel**2,       # v
#             (np.deg2rad(5))**2,  # psi noise
#             (std_turn)**2,    # omega
#             (1e-2)**2,        # a
#             std_size**2       # h
#         ])

#         cov_pred = F @ covariance @ F.T + Q
#         return mean_pred, cov_pred

#     def project(self, mean, covariance):
#         H = self._H_jacobian(mean)
#         z_pred = self._h(mean)

#         std_pos = self._std_weight_pos * mean[6]
#         std_size = self._std_weight_size * mean[6]

#         R = np.diag([
#             std_pos**2,  # px
#             std_pos**2,  # py
#             (1e-1)**2,   # a
#             std_size**2  # h
#         ])
#         S = H @ covariance @ H.T + R
#         return z_pred, S, H, R

#     def update(self, mean, covariance, measurement):
#         z_pred, S, H, _ = self.project(mean, covariance)
#         y = measurement - z_pred

#         # Solve for K with Cholesky for stability
#         chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
#         K = scipy.linalg.cho_solve((chol, lower), (covariance @ H.T).T,
#                                    check_finite=False).T

#         mean_upd = mean + K @ y
#         cov_upd = covariance - K @ S @ K.T
#         return mean_upd, cov_upd

#     def gating_distance(self, mean, covariance, measurements, only_position=False):
#         z_pred, S, H, _ = self.project(mean, covariance)
#         if only_position:
#             H = H[:2, :]
#             z_pred = z_pred[:2]
#             S = S[:2, :2]
#             measurements = measurements[:, :2]

#         chol_factor = np.linalg.cholesky(S)
#         d = measurements - z_pred
#         z = scipy.linalg.solve_triangular(chol_factor, d.T,
#                                           lower=True, check_finite=False, overwrite_b=True)
#         sq_maha = np.sum(z * z, axis=0)
#         return sq_maha

def wrap_angle(a):
    """[-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi

class ExtendedKalmanFilter:
    """
    CTRV EKF for bbox tracking.

    State x = [px, py, v, psi, omega, a, h]
    Meas  z = [px, py, a, h]
    """

    def __init__(self, dt=1.0,
                 std_pos=1/20, std_size=1/20,
                 std_vel=1/10, std_turn=1/30,
                 min_h=5., max_h=2000.,
                 min_a=0.05, max_a=10.,
                 max_v=2000., max_omega=np.deg2rad(45)):
        self.dt = dt
        self.ndim = 7
        self.zdim = 4

        # scale factors (настраиваемые)
        self._std_weight_pos  = std_pos
        self._std_weight_size = std_size
        self._std_weight_vel  = std_vel
        self._std_weight_turn = std_turn

        # клиппинги
        self._min_h, self._max_h = min_h, max_h
        self._min_a, self._max_a = min_a, max_a
        self._max_v, self._max_omega = max_v, max_omega

    # ---------- motion model ----------
    def _f(self, x):
        px, py, v, psi, omega, a, h = x
        dt = self.dt

        w = omega
        psi_dt = psi + w * dt

        # series expansion if |w| small
        if np.abs(w) < 1e-6:
            s = v * dt
            px_n = px + s * np.cos(psi)
            py_n = py + s * np.sin(psi)
        else:
            px_n = px + (v / w) * (np.sin(psi_dt) - np.sin(psi))
            py_n = py - (v / w) * (np.cos(psi_dt) - np.cos(psi))

        v_n = v
        psi_n = wrap_angle(psi_dt)
        omega_n = w
        a_n = a
        h_n = h
        return np.array([px_n, py_n, v_n, psi_n, omega_n, a_n, h_n], dtype=float)

    def _F_jacobian(self, x):
        px, py, v, psi, omega, a, h = x
        dt = self.dt
        F = np.eye(self.ndim)

        w = omega
        sp, cp = np.sin(psi), np.cos(psi)

        if np.abs(w) < 1e-6:
            # linearised straight motion
            F[0, 2] = cp * dt
            F[0, 3] = -v * sp * dt
            F[1, 2] = sp * dt
            F[1, 3] = v * cp * dt
        else:
            psi_dt = psi + w * dt
            sp2, cp2 = np.sin(psi_dt), np.cos(psi_dt)

            F[0, 2] = (sp2 - sp) / w
            F[0, 3] = (v / w) * (cp2 - cp)
            F[0, 4] = (v * (w * cp2 * dt) - v * (sp2 - sp)) / (w**2)

            F[1, 2] = -(cp2 - cp) / w
            F[1, 3] = (v / w) * (sp2 - sp)
            F[1, 4] = (v * (w * sp2 * dt) - v * (cp2 - cp)) / (w**2)

            F[3, 4] = dt  # d psi / d omega

        return F

    # ---------- measurement model ----------
    def _h(self, x):
        return np.array([x[0], x[1], x[5], x[6]], dtype=float)

    def _H_jacobian(self, x):
        H = np.zeros((self.zdim, self.ndim))
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 5] = 1.0
        H[3, 6] = 1.0
        return H

    # ---------- API ----------
    def initiate(self, measurement):
        px, py, a, h = measurement
        v = 0.0
        psi = 0.0
        omega = 0.0
        mean = np.array([px, py, v, psi, omega, a, h], dtype=float)

        std = [
            2 * self._std_weight_pos  * h,  # px
            2 * self._std_weight_pos  * h,  # py
            5 * self._std_weight_vel  * h,  # v
            np.deg2rad(10),                # psi
            np.deg2rad(5),                 # omega
            1e-2,                          # a
            2 * self._std_weight_size * h  # h
        ]
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean, covariance):
        F = self._F_jacobian(mean)
        mean = self._f(mean)

        # process noise
        h = np.clip(mean[6], self._min_h, self._max_h)
        std_pos  = self._std_weight_pos  * h
        std_vel  = self._std_weight_vel  * h
        std_turn = self._std_weight_turn
        std_size = self._std_weight_size * h

        Q = np.diag([
            std_pos**2,               # px
            std_pos**2,               # py
            std_vel**2,               # v
            (np.deg2rad(5))**2,       # psi
            std_turn**2,              # omega
            (1e-2)**2,                # a
            std_size**2               # h
        ])

        covariance = F @ covariance @ F.T + Q

        mean = self._clip_state(mean)
        covariance = 0.5 * (covariance + covariance.T)  # symmetrize
        return mean, covariance

    def project(self, mean, covariance):
        H = self._H_jacobian(mean)
        z_pred = self._h(mean)

        h = np.clip(mean[6], self._min_h, self._max_h)
        std_pos  = self._std_weight_pos  * h
        std_size = self._std_weight_size * h

        R = np.diag([
            std_pos**2,
            std_pos**2,
            (1e-1)**2,
            std_size**2
        ])

        S = H @ covariance @ H.T + R
        return z_pred, S, H, R

    def update(self, mean, covariance, measurement):
        z_pred, S, H, _ = self.project(mean, covariance)
        y = measurement - z_pred

        chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol, lower), (covariance @ H.T).T,
                         check_finite=False).T

        mean = mean + K @ y
        covariance = covariance - K @ S @ K.T

        mean = self._clip_state(mean)
        covariance = 0.5 * (covariance + covariance.T)
        return mean, covariance

    def _clip_state(self, x):
        x[2] = np.clip(x[2], -self._max_v, self._max_v)              # v
        x[4] = np.clip(x[4], -self._max_omega, self._max_omega)      # omega
        x[5] = np.clip(x[5], self._min_a, self._max_a)               # a
        x[6] = np.clip(x[6], self._min_h, self._max_h)               # h
        x[3] = wrap_angle(x[3])                                      # psi
        return x

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        z_pred, S, H, _ = self.project(mean, covariance)
        if only_position:
            H = H[:2, :]
            z_pred = z_pred[:2]
            S = S[:2, :2]
            measurements = measurements[:, :2]

        chol = scipy.linalg.cholesky(S, lower=True, check_finite=False)
        d = measurements - z_pred
        z = scipy.linalg.solve_triangular(chol, d.T, lower=True,
                                check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)

# ---------- utils ----------
def _num_jacobian(f, x, eps=1e-5):
    """Центральная разностная аппроксимация Якобиана (лучше, чем односторонняя)."""
    n = x.shape[0]
    F = np.zeros((n, n), dtype=float)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps
        F[:, i] = (f(x + dx) - f(x - dx)) / (2 * eps)
    return F

def _symmetrize(P):
    return 0.5 * (P + P.T)

# ---------- EKF ----------
class ExtendedKalmanFilterAccel:
    """
    EKF: Constant Turn Rate + linear Acceleration along heading.
    state x = [px, py, v, psi, omega, a_lin, a, h]
    meas  z = [px, py, a, h]
    """

    def __init__(self, dt=1.0,
                 std_pos = 1/20,   # относит. шум позиции   (высота * coef)
                 std_size= 1/20,   # относит. шум размера   (высота * coef)
                 std_vel = 1/10,   # относит. шум скорости   (высота * coef)
                 std_turn= 1/30,   # абсолютн. шум угл.скорости (рад/с)
                 std_acc = 1/10,   # относит. шум лин.ускорения (высота * coef)
                 jacobian_eps=1e-5,
                 clip_cfg=None):
        self.dt = dt
        self.ndim = 8
        self.zdim = 4

        self._std_weight_pos  = std_pos
        self._std_weight_size = std_size
        self._std_weight_vel  = std_vel
        self._std_weight_turn = std_turn
        self._std_weight_acc  = std_acc

        self._jacobian_eps = jacobian_eps

        # лимиты на состояние
        if clip_cfg is None:
            clip_cfg = dict(v_max=2000., a_max=200.,
                            a_min=0.05, a_max_ratio=10.,
                            h_min=5.,   h_max=2000.)
        self.clip_cfg = clip_cfg

    # ---------- motion model ----------
    def _f(self, x):
        px, py, v, psi, omega, a_lin, a, h = x
        dt  = self.dt
        eps = 1e-6

        v_new   = v + a_lin * dt
        psi_new = psi + omega * dt

        if abs(omega) < eps:
            ds = v * dt + 0.5 * a_lin * dt**2
            px_new = px + ds * np.cos(psi)
            py_new = py + ds * np.sin(psi)
        else:
            v_avg  = 0.5 * (v + v_new)
            px_new = px + (v_avg / omega) * (np.sin(psi_new) - np.sin(psi))
            py_new = py - (v_avg / omega) * (np.cos(psi_new) - np.cos(psi))

        return np.array([px_new, py_new, v_new, psi_new, omega, a_lin, a, h], dtype=float)

    def _F_jacobian(self, x):
        return _num_jacobian(self._f, x, self._jacobian_eps)

    # ---------- measurement model ----------
    def _h(self, x):
        return np.array([x[0], x[1], x[6], x[7]], dtype=float)

    def _H_jacobian(self, x):
        H = np.zeros((self.zdim, self.ndim), dtype=float)
        H[0, 0] = 1.0  # px
        H[1, 1] = 1.0  # py
        H[2, 6] = 1.0  # a
        H[3, 7] = 1.0  # h
        return H

    # ---------- helpers ----------
    def _clip_state(self, x):
        cfg = self.clip_cfg
        x[2] = np.clip(x[2], -cfg['v_max'],    cfg['v_max'])      # v
        x[5] = np.clip(x[5], -cfg['a_max'],    cfg['a_max'])      # a_lin
        x[6] = np.clip(x[6],  cfg['a_min'],    cfg['a_max_ratio'])# aspect ratio
        x[7] = np.clip(x[7],  cfg['h_min'],    cfg['h_max'])      # height
        return x

    def _process_noise(self, h):
        std_pos  = self._std_weight_pos  * h
        std_size = self._std_weight_size * h
        std_vel  = self._std_weight_vel  * h
        std_acc  = self._std_weight_acc  * h
        std_turn = self._std_weight_turn

        Q = np.diag([
            std_pos**2,             # px
            std_pos**2,             # py
            std_vel**2,             # v
            (np.deg2rad(5))**2,     # psi
            std_turn**2,            # omega
            std_acc**2,             # a_lin
            (1e-2)**2,              # a
            std_size**2             # h
        ])
        return Q

    def _meas_noise(self, h):
        std_pos  = self._std_weight_pos  * h
        std_size = self._std_weight_size * h
        R = np.diag([
            std_pos**2,
            std_pos**2,
            (1e-1)**2,      # аспект-отношение точнее обычно
            std_size**2
        ])
        return R

    # ---------- API ----------
    def initiate(self, measurement):
        px, py, a, h = measurement
        mean = np.array([px, py, 0.0, 0.0, 0.0, 0.0, a, h], dtype=float)

        std = [
            2 * self._std_weight_pos  * h,   # px
            2 * self._std_weight_pos  * h,   # py
            2 * self._std_weight_vel  * h,   # v
            np.deg2rad(10),                  # psi
            np.deg2rad(5),                   # omega
            2 * self._std_weight_acc  * h,   # a_lin
            1e-2,                            # a
            2 * self._std_weight_size * h    # h
        ]
        cov = np.diag(np.square(std))
        return mean, cov

    def predict(self, mean, covariance):
        F         = self._F_jacobian(mean)
        mean_pred = self._f(mean)
        Q         = self._process_noise(mean[7])

        cov_pred  = F @ covariance @ F.T + Q
        cov_pred  = _symmetrize(cov_pred)

        mean_pred = self._clip_state(mean_pred)
        return mean_pred, cov_pred

    def project(self, mean, covariance):
        H      = self._H_jacobian(mean)
        z_pred = self._h(mean)
        R      = self._meas_noise(mean[7])

        S = H @ covariance @ H.T + R
        S = _symmetrize(S)
        return z_pred, S, H, R

    def update(self, mean, covariance, measurement):
        z_pred, S, H, _ = self.project(mean, covariance)
        y = measurement - z_pred

        chol, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol, lower), (covariance @ H.T).T,
                         check_finite=False).T

        mean_upd = mean + K @ y
        cov_upd  = covariance - K @ S @ K.T
        cov_upd  = _symmetrize(cov_upd)

        mean_upd = self._clip_state(mean_upd)
        return mean_upd, cov_upd

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        z_pred, S, H, _ = self.project(mean, covariance)
        if only_position:
            H = H[:2, :]
            z_pred = z_pred[:2]
            S = S[:2, :2]
            measurements = measurements[:, :2]

        chol = scipy.linalg.cholesky(S, lower=True, check_finite=False)
        d = measurements - z_pred
        z = scipy.linalg.solve_triangular(chol, d.T, lower=True,
                                check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)

from threading import Thread
import math

class DummyThread(Thread):
    def run(self):
        while True:
            pass


def quat2rpy(q0, q1, q2, q3):
    r = math.atan2(2*(q0*q1+q2*q3), 1-2*(q1*q1+q2*q2))*180/math.pi
    p = math.asin(2*(q0*q2-q1*q3))*180/math.pi
    y = math.atan2(2*(q0*q3+q1*q2), 1-2*(q2*q2+q3*q3))*180/math.pi
    return (r, p, y)


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v


def qv_mult(q1, v):
    q2 = (0.0,) + v
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]


def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z


def q_conjugate(q):
    q = normalize(q)
    w, x, y, z = q
    return w, -x, -y, -z

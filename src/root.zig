const std = @import("std");
const math = @import("std").math;

pub fn Vec2(T: type) type {
    return struct {
        const Self = @This();

        x: T,
        y: T,

        pub inline fn new(x: T, y: T) Self {
            return .{ .x = x, .y = y };
        }

        pub inline fn zero() Self {
            return .{ .x = 0, .y = 0 };
        }

        pub inline fn one(s: T) Self {
            return .{ .x = s, .y = s };
        }

        pub inline fn eql(v1: Self, v2: Self) bool {
            return (v1.x == v2.x and v1.y == v2.y);
        }

        pub inline fn xyz(v: Self) Vec3(T) {
            return .{ .x = v.x, .y = v.y, .z = 0 };
        }

        pub inline fn toArray(v: Self) [2]T {
            return .{ v.x, v.y };
        }

        pub inline fn add(v1: Self, v2: Self) Self {
            return .{ .x = v1.x + v2.x, .y = v1.y + v2.y };
        }

        pub inline fn sub(v1: Self, v2: Self) Self {
            return .{ .x = v1.x - v2.x, .y = v1.y - v2.y };
        }

        pub inline fn mul(v: Self, s: T) Self {
            return .{ .x = v.x * s, .y = v.y * s };
        }

        pub inline fn div(v: Self, s: T) Self {
            return .{ .x = v.x / s, .y = v.y / s };
        }

        pub inline fn negate(v: Self) Self {
            return .{ .x = -v.x, .y = -v.y };
        }

        pub inline fn abs(v: Self) Self {
            return .{ .x = @abs(v.x), .y = @abs(v.y) };
        }

        pub inline fn floor(v: Self) Self {
            return .{
                .x = @floor(v.x),
                .y = @floor(v.y),
            };
        }

        pub inline fn clamp(v: Self, lower: Self, upper: Self) Self {
            return .{
                .x = math.clamp(v.x, lower.x, upper.x),
                .y = math.clamp(v.y, lower.y, upper.y),
            };
        }

        pub inline fn dot(v1: Self, v2: Self) T {
            return v1.x * v2.x + v1.y * v2.y;
        }

        pub inline fn lengthSquared(v: Self) T {
            return v.dot(v);
        }

        pub inline fn length(v: Self) T {
            return @sqrt(lengthSquared(v));
        }

        pub inline fn normalize(v: Self) Self {
            const len = length(v);
            return if (len != 0) div(v, len) else zero();
        }

        pub inline fn distance(v1: Self, v2: Self) T {
            return length(sub(v1, v2));
        }

        pub inline fn distanceSquared(v1: Self, v2: Self) T {
            return lengthSquared(sub(v1, v2));
        }

        pub inline fn min(a: Self, b: Self) Self {
            return .{
                .x = @min(a.x, b.x),
                .y = @min(a.y, b.y),
            };
        }

        pub inline fn max(a: Self, b: Self) Self {
            return .{
                .x = @max(a.x, b.x),
                .y = @max(a.y, b.y),
            };
        }

        pub inline fn lerp(a: Self, b: Self, t: T) Self {
            return .{
                .x = math.lerp(a.x, b.x, t),
                .y = math.lerp(a.y, b.y, t),
            };
        }

        pub inline fn perpendicular(v: Self) Self {
            // Rotated 90° counter-clockwise
            return .{ .x = -v.y, .y = v.x };
        }

        pub inline fn fromAngle(theta: T) Self {
            return .{ .x = @cos(theta), .y = @sin(theta) };
        }

        pub inline fn angle(v: Self) T {
            return math.atan2(v.y, v.x);
        }

        pub inline fn isZero(v: Self) bool {
            return v.x == 0 and v.y == 0;
        }
    };
}

pub fn Vec3(T: type) type {
    return struct {
        const Self = @This();

        x: T,
        y: T,
        z: T,

        pub inline fn new(x: T, y: T, z: T) Self {
            return .{ .x = x, .y = y, .z = z };
        }

        pub inline fn zero() Self {
            return .{ .x = 0, .y = 0, .z = 0 };
        }

        pub inline fn one(s: T) Self {
            return .{ .x = s, .y = s, .z = s };
        }

        pub inline fn eql(v1: Self, v2: Self) bool {
            return (v1.x == v2.x and v1.y == v2.y and v1.z == v2.z);
        }

        pub inline fn xy(v: Self) Vec2(T) {
            return .{ .x = v.x, .y = v.y };
        }

        pub inline fn yz(v: Self) Vec2(T) {
            return .{ .x = v.y, .y = v.z };
        }

        pub inline fn xz(v: Self) Vec2(T) {
            return .{ .x = v.x, .y = v.z };
        }

        pub inline fn toArray(v: Self) [3]T {
            return .{ v.x, v.y, v.z };
        }

        pub inline fn add(v1: Self, v2: Self) Self {
            return .{
                .x = v1.x + v2.x,
                .y = v1.y + v2.y,
                .z = v1.z + v2.z,
            };
        }

        pub inline fn sub(v1: Self, v2: Self) Self {
            return .{
                .x = v1.x - v2.x,
                .y = v1.y - v2.y,
                .z = v1.z - v2.z,
            };
        }

        pub inline fn mul(v: Self, s: T) Self {
            return .{
                .x = v.x * s,
                .y = v.y * s,
                .z = v.z * s,
            };
        }

        pub inline fn div(v: Self, s: T) Self {
            return .{
                .x = v.x / s,
                .y = v.y / s,
                .z = v.z / s,
            };
        }

        pub inline fn negate(v: Self) Self {
            return .{
                .x = -v.x,
                .y = -v.y,
                .z = -v.z,
            };
        }

        pub inline fn abs(v: Self) Self {
            return .{
                .x = @abs(v.x),
                .y = @abs(v.y),
                .z = @abs(v.z),
            };
        }

        pub inline fn floor(v: Self) Self {
            return .{
                .x = @floor(v.x),
                .y = @floor(v.y),
                .z = @floor(v.z),
            };
        }

        pub inline fn clamp(v: Self, lower: Self, upper: Self) Self {
            return .{
                .x = math.clamp(v.x, lower.x, upper.x),
                .y = math.clamp(v.y, lower.y, upper.y),
                .z = math.clamp(v.z, lower.z, upper.z),
            };
        }

        pub inline fn dot(v1: Self, v2: Self) T {
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
        }

        pub inline fn cross(v1: Self, v2: Self) Self {
            return .{
                .x = v1.y * v2.z - v1.z * v2.y,
                .y = v1.z * v2.x - v1.x * v2.z,
                .z = v1.x * v2.y - v1.y * v2.x,
            };
        }

        pub inline fn lengthSquared(v: Self) T {
            return dot(v, v);
        }

        pub inline fn length(v: Self) T {
            return @sqrt(lengthSquared(v));
        }

        pub inline fn normalize(v: Self) Self {
            const len = length(v);
            return if (len != 0) div(v, len) else zero();
        }

        pub inline fn distance(v1: Self, v2: Self) T {
            return length(sub(v1, v2));
        }

        pub inline fn distanceSquared(v1: Self, v2: Self) T {
            return lengthSquared(sub(v1, v2));
        }

        pub inline fn min(a: Self, b: Self) Self {
            return .{
                .x = @min(a.x, b.x),
                .y = @min(a.y, b.y),
                .z = @min(a.z, b.z),
            };
        }

        pub inline fn max(a: Self, b: Self) Self {
            return .{
                .x = @max(a.x, b.x),
                .y = @max(a.y, b.y),
                .z = @max(a.z, b.z),
            };
        }

        pub inline fn lerp(a: Self, b: Self, t: T) Self {
            return .{
                .x = math.lerp(a.x, b.x, t),
                .y = math.lerp(a.y, b.y, t),
                .z = math.lerp(a.z, b.z, t),
            };
        }

        pub inline fn project(v: Self, onto: Self) Self {
            // projection of v onto "onto"
            const scale = dot(v, onto) / lengthSquared(onto);
            return mul(onto, scale);
        }

        pub inline fn isZero(v: Self) bool {
            return v.x == 0 and v.y == 0 and v.z == 0;
        }

        pub inline fn equalsApprox(a: Self, b: Self, epsilon: T) bool {
            return @abs(a.x - b.x) <= epsilon and
                @abs(a.y - b.y) <= epsilon and
                @abs(a.z - b.z) <= epsilon;
        }
    };
}

pub fn Vec4(T: type) type {
    return struct {
        const Self = @This();

        x: T,
        y: T,
        z: T,
        w: T,

        pub inline fn new(x: T, y: T, z: T, w: T) Self {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }

        pub inline fn zero() Self {
            return .{ .x = 0, .y = 0, .z = 0, .w = 0 };
        }

        pub inline fn one(s: T) Self {
            return .{ .x = s, .y = s, .z = s, .w = s };
        }

        pub inline fn eql(v1: Self, v2: Self) bool {
            return v1.x == v2.x and v1.y == v2.y and v1.z == v2.z and v1.w == v2.w;
        }

        pub inline fn xyz(v: Self) Vec3(T) {
            return .{ .x = v.x, .y = v.y, .z = v.z };
        }

        pub inline fn xy(v: Self) Vec2(T) {
            return .{ .x = v.x, .y = v.y };
        }

        pub inline fn zw(v: Self) Vec2(T) {
            return .{ .x = v.z, .y = v.w };
        }

        pub inline fn toArray(v: Self) [4]T {
            return .{ v.x, v.y, v.z, v.w };
        }

        pub inline fn add(v1: Self, v2: Self) Self {
            return .{
                .x = v1.x + v2.x,
                .y = v1.y + v2.y,
                .z = v1.z + v2.z,
                .w = v1.w + v2.w,
            };
        }

        pub inline fn sub(v1: Self, v2: Self) Self {
            return .{
                .x = v1.x - v2.x,
                .y = v1.y - v2.y,
                .z = v1.z - v2.z,
                .w = v1.w - v2.w,
            };
        }

        pub inline fn mul(v: Self, s: T) Self {
            return .{
                .x = v.x * s,
                .y = v.y * s,
                .z = v.z * s,
                .w = v.w * s,
            };
        }

        pub inline fn div(v: Self, s: T) Self {
            return .{
                .x = v.x / s,
                .y = v.y / s,
                .z = v.z / s,
                .w = v.w / s,
            };
        }

        pub inline fn negate(v: Self) Self {
            return .{
                .x = -v.x,
                .y = -v.y,
                .z = -v.z,
                .w = -v.w,
            };
        }

        pub inline fn abs(v: Self) Self {
            return .{
                .x = @abs(v.x),
                .y = @abs(v.y),
                .z = @abs(v.z),
                .w = @abs(v.w),
            };
        }

        pub inline fn floor(v: Self) Self {
            return .{
                .x = @floor(v.x),
                .y = @floor(v.y),
                .z = @floor(v.z),
                .w = @floor(v.w),
            };
        }

        pub inline fn clamp(v: Self, lower: Self, upper: Self) Self {
            return .{
                .x = math.clamp(v.x, lower.x, upper.x),
                .y = math.clamp(v.y, lower.y, upper.y),
                .z = math.clamp(v.z, lower.z, upper.z),
                .w = math.clamp(v.w, lower.w, upper.w),
            };
        }

        pub inline fn dot(v1: Self, v2: Self) T {
            return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
        }

        pub inline fn lengthSquared(v: Self) T {
            return dot(v, v);
        }

        pub inline fn length(v: Self) T {
            return @sqrt(lengthSquared(v));
        }

        pub inline fn normalize(v: Self) Self {
            const len = length(v);
            return if (len != 0) div(v, len) else zero();
        }

        pub inline fn distance(v1: Self, v2: Self) T {
            return length(sub(v1, v2));
        }

        pub inline fn distanceSquared(v1: Self, v2: Self) T {
            return lengthSquared(sub(v1, v2));
        }

        pub inline fn min(a: Self, b: Self) Self {
            return .{
                .x = @min(a.x, b.x),
                .y = @min(a.y, b.y),
                .z = @min(a.z, b.z),
                .w = @min(a.w, b.w),
            };
        }

        pub inline fn max(a: Self, b: Self) Self {
            return .{
                .x = @max(a.x, b.x),
                .y = @max(a.y, b.y),
                .z = @max(a.z, b.z),
                .w = @max(a.w, b.w),
            };
        }

        pub inline fn lerp(a: Self, b: Self, t: T) Self {
            return .{
                .x = math.lerp(a.x, b.x, t),
                .y = math.lerp(a.y, b.y, t),
                .z = math.lerp(a.z, b.z, t),
                .w = math.lerp(a.w, b.w, t),
            };
        }

        pub inline fn isZero(v: Self) bool {
            return v.x == 0 and v.y == 0 and v.z == 0 and v.w == 0;
        }

        pub inline fn equalsApprox(a: Self, b: Self, epsilon: T) bool {
            return @abs(a.x - b.x) <= epsilon and
                @abs(a.y - b.y) <= epsilon and
                @abs(a.z - b.z) <= epsilon and
                @abs(a.w - b.w) <= epsilon;
        }
    };
}

pub fn Quat(T: type) type {
    return struct {
        const Self = @This();

        x: T,
        y: T,
        z: T,
        w: T,

        pub inline fn new(x: T, y: T, z: T, w: T) Self {
            return .{ .x = x, .y = y, .z = z, .w = w };
        }

        pub inline fn identity() Self {
            return .{ .x = 0, .y = 0, .z = 0, .w = 1 };
        }

        pub inline fn fromAxisAngle(axis: Vec3(T), angle: T) Self {
            const half = angle / 2;
            const s = @sin(half);
            return .{
                .x = axis.x * s,
                .y = axis.y * s,
                .z = axis.z * s,
                .w = @cos(half),
            };
        }

        pub inline fn fromEuler(pitch: T, yaw: T, roll: T) Self {
            const cy = @cos(yaw * 0.5);
            const sy = @sin(yaw * 0.5);
            const cp = @cos(pitch * 0.5);
            const sp = @sin(pitch * 0.5);
            const cr = @cos(roll * 0.5);
            const sr = @sin(roll * 0.5);

            return .{
                .x = sr * cp * cy - cr * sp * sy,
                .y = cr * sp * cy + sr * cp * sy,
                .z = cr * cp * sy - sr * sp * cy,
                .w = cr * cp * cy + sr * sp * sy,
            };
        }

        pub inline fn normalize(q: Self) Self {
            const len = @sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
            return if (len != 0) .{
                .x = q.x / len,
                .y = q.y / len,
                .z = q.z / len,
                .w = q.w / len,
            } else identity();
        }

        pub inline fn conjugate(q: Self) Self {
            return .{ .x = -q.x, .y = -q.y, .z = -q.z, .w = q.w };
        }

        pub inline fn inverse(q: Self) Self {
            const len_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
            return if (len_sq != 0) .{
                .x = -q.x / len_sq,
                .y = -q.y / len_sq,
                .z = -q.z / len_sq,
                .w = q.w / len_sq,
            } else identity();
        }

        pub inline fn multiply(a: Self, b: Self) Self {
            return .{
                .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            };
        }

        pub inline fn rotate(q: Self, v: Vec3(T)) Vec3(T) {
            const qv = Vec3(T).new(q.x, q.y, q.z);
            const uv = Vec3(T).cross(qv, v);
            const uuv = Vec3(T).cross(qv, uv);
            const uv_scaled = Vec3(T).mul(uv, q.w * 2);
            const uuv_scaled = Vec3(T).mul(uuv, 2);
            return Vec3(T).add(v, Vec3(T).add(uv_scaled, uuv_scaled));
        }

        pub inline fn toMatrix(q: Self) [3][3]T {
            const x2 = q.x + q.x;
            const y2 = q.y + q.y;
            const z2 = q.z + q.z;

            const xx = q.x * x2;
            const yy = q.y * y2;
            const zz = q.z * z2;
            const xy = q.x * y2;
            const xz = q.x * z2;
            const yz = q.y * z2;
            const wx = q.w * x2;
            const wy = q.w * y2;
            const wz = q.w * z2;

            return .{
                .{ 1 - (yy + zz), xy - wz, xz + wy },
                .{ xy + wz, 1 - (xx + zz), yz - wx },
                .{ xz - wy, yz + wx, 1 - (xx + yy) },
            };
        }

        pub inline fn slerp(a: Self, b: Self, t: T) Self {
            var cos_theta = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            var b_ = b;

            if (cos_theta < 0) {
                b_ = .{ .x = -b.x, .y = -b.y, .z = -b.z, .w = -b.w };
                cos_theta = -cos_theta;
            }

            if (cos_theta > 0.9995) {
                // Linear interpolation
                return normalize(.{
                    .x = a.x + t * (b_.x - a.x),
                    .y = a.y + t * (b_.y - a.y),
                    .z = a.z + t * (b_.z - a.z),
                    .w = a.w + t * (b_.w - a.w),
                });
            }

            const angle = math.acos(cos_theta);
            const sin_theta = @sin(angle);
            const ta = @sin((1 - t) * angle) / sin_theta;
            const tb = @sin(t * angle) / sin_theta;

            return .{
                .x = a.x * ta + b_.x * tb,
                .y = a.y * ta + b_.y * tb,
                .z = a.z * ta + b_.z * tb,
                .w = a.w * ta + b_.w * tb,
            };
        }

        pub inline fn equalsApprox(a: Self, b: Self, epsilon: T) bool {
            return @abs(a.x - b.x) <= epsilon and
                @abs(a.y - b.y) <= epsilon and
                @abs(a.z - b.z) <= epsilon and
                @abs(a.w - b.w) <= epsilon;
        }
    };
}

pub fn Mat2(T: type) type {
    return struct {
        const Self = @This();
        cols: [2][2]T, // column-major layout

        pub fn identity() Self {
            return .{ .cols = .{
                .{ 1, 0 },
                .{ 0, 1 },
            } };
        }

        pub fn mul(a: Self, b: Self) Self {
            var out: Self = undefined;
            for (0..2) |i| {
                for (0..2) |j| {
                    out.cols[i][j] =
                        a.cols[0][j] * b.cols[i][0] +
                        a.cols[1][j] * b.cols[i][1];
                }
            }
            return out;
        }

        pub fn transpose(m: Self) Self {
            return .{ .cols = .{
                .{ m.cols[0][0], m.cols[1][0] },
                .{ m.cols[0][1], m.cols[1][1] },
            } };
        }
    };
}

pub fn Mat3(T: type) type {
    return struct {
        const Self = @This();
        cols: [3][3]T, // column-major layout

        pub fn identity() Self {
            return .{ .cols = .{
                .{ 1, 0, 0 },
                .{ 0, 1, 0 },
                .{ 0, 0, 1 },
            } };
        }

        pub fn mul(a: Self, b: Self) Self {
            var out: Self = undefined;
            for (0..3) |i| {
                for (0..3) |j| {
                    out.cols[i][j] =
                        a.cols[0][j] * b.cols[i][0] +
                        a.cols[1][j] * b.cols[i][1] +
                        a.cols[2][j] * b.cols[i][2];
                }
            }
            return out;
        }

        pub fn transpose(m: Self) Self {
            return .{ .cols = .{
                .{ m.cols[0][0], m.cols[1][0], m.cols[2][0] },
                .{ m.cols[0][1], m.cols[1][1], m.cols[2][1] },
                .{ m.cols[0][2], m.cols[1][2], m.cols[2][2] },
            } };
        }
    };
}

pub fn Mat4(T: type) type {
    return struct {
        const Self = @This();
        cols: [4][4]T, // column-major

        pub fn identity() Self {
            return .{ .cols = .{
                .{ 1, 0, 0, 0 },
                .{ 0, 1, 0, 0 },
                .{ 0, 0, 1, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        pub fn translation(v: Vec3) Self {
            var m = Self.identity();
            m.cols[3][0] = v.x;
            m.cols[3][1] = v.y;
            m.cols[3][2] = v.z;
            return m;
        }

        pub fn scaling(v: Vec3) Self {
            var m = Self.identity();
            m.cols[0][0] = v.x;
            m.cols[1][1] = v.y;
            m.cols[2][2] = v.z;
            return m;
        }

        pub fn rotationX(angle: T) Self {
            const c = std.math.cos(angle);
            const s = std.math.sin(angle);
            return .{ .cols = .{
                .{ 1, 0, 0, 0 },
                .{ 0, c, s, 0 },
                .{ 0, -s, c, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        pub fn rotationY(angle: T) Self {
            const c = std.math.cos(angle);
            const s = std.math.sin(angle);
            return .{ .cols = .{
                .{ c, 0, -s, 0 },
                .{ 0, 1, 0, 0 },
                .{ s, 0, c, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        pub fn rotationZ(angle: T) Self {
            const c = std.math.cos(angle);
            const s = std.math.sin(angle);
            return .{ .cols = .{
                .{ c, s, 0, 0 },
                .{ -s, c, 0, 0 },
                .{ 0, 0, 1, 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        pub fn lookAt(eye: Vec3(T), center: Vec3(T), up: Vec3(T)) Self {
            const forward = center.sub(eye).normalize();
            const right = forward.cross(up).normalize();
            const newUp = right.cross(forward);

            return .{ .cols = .{
                .{ right.x, newUp.x, -forward.x, 0 },
                .{ right.y, newUp.y, -forward.y, 0 },
                .{ right.z, newUp.z, -forward.z, 0 },
                .{ -right.dot(eye), -newUp.dot(eye), forward.dot(eye), 1 },
            } };
        }

        pub fn perspective(fovy: T, aspect: T, near: T, far: T) Self {
            const f = 1 / @tan(fovy / 2);
            return .{ .cols = .{
                .{ f / aspect, 0, 0, 0 },
                .{ 0, f, 0, 0 },
                .{ 0, 0, (far + near) / (near - far), -1 },
                .{ 0, 0, (2 * far * near) / (near - far), 0 },
            } };
        }

        pub fn orthographic(left: T, right: T, bottom: T, top: T, near: T, far: T) Self {
            const rl = right - left;
            const tb = top - bottom;
            const @"fn" = far - near;

            return .{ .cols = .{
                .{ 2 / rl, 0, 0, 0 },
                .{ 0, 2 / tb, 0, 0 },
                .{ 0, 0, -2 / @"fn", 0 },
                .{ -(right + left) / rl, -(top + bottom) / tb, -(far + near) / @"fn", 1 },
            } };
        }

        pub fn frustum(left: T, right: T, bottom: T, top: T, near: T, far: T) Self {
            const rl = right - left;
            const tb = top - bottom;
            const @"fn" = far - near;

            return .{ .cols = .{
                .{ (2 * near) / rl, 0, 0, 0 },
                .{ 0, (2 * near) / tb, 0, 0 },
                .{ (right + left) / rl, (top + bottom) / tb, -(far + near) / @"fn", -1 },
                .{ 0, 0, -(2 * far * near) / @"fn", 0 },
            } };
        }

        pub fn mul(a: Self, b: Self) Self {
            var out: Self = undefined;
            for (0..4) |i| {
                for (0..4) |j| {
                    out.cols[i][j] =
                        a.cols[0][j] * b.cols[i][0] +
                        a.cols[1][j] * b.cols[i][1] +
                        a.cols[2][j] * b.cols[i][2] +
                        a.cols[3][j] * b.cols[i][3];
                }
            }
            return out;
        }

        pub fn transpose(m: Self) Self {
            return .{ .cols = .{
                .{ m.cols[0][0], m.cols[1][0], m.cols[2][0], m.cols[3][0] },
                .{ m.cols[0][1], m.cols[1][1], m.cols[2][1], m.cols[3][1] },
                .{ m.cols[0][2], m.cols[1][2], m.cols[2][2], m.cols[3][2] },
                .{ m.cols[0][3], m.cols[1][3], m.cols[2][3], m.cols[3][3] },
            } };
        }

        pub fn transform(m: Self, v: Vec4) Vec4 {
            return .{
                .x = m.cols[0][0] * v.x + m.cols[1][0] * v.y + m.cols[2][0] * v.z + m.cols[3][0] * v.w,
                .y = m.cols[0][1] * v.x + m.cols[1][1] * v.y + m.cols[2][1] * v.z + m.cols[3][1] * v.w,
                .z = m.cols[0][2] * v.x + m.cols[1][2] * v.y + m.cols[2][2] * v.z + m.cols[3][2] * v.w,
                .w = m.cols[0][3] * v.x + m.cols[1][3] * v.y + m.cols[2][3] * v.z + m.cols[3][3] * v.w,
            };
        }

        pub fn inverse(m: Self) ?Self {
            // Using the Gauss-Jordan elimination or analytic formula is complex,
            // but here is a common 4x4 inverse using cofactors & determinants.

            var inv: [16]T = undefined;
            const a = m.cols;

            inv[0] = a[1][1] * a[2][2] * a[3][3] -
                a[1][1] * a[3][2] * a[2][3] -
                a[2][1] * a[1][2] * a[3][3] +
                a[2][1] * a[3][2] * a[1][3] +
                a[3][1] * a[1][2] * a[2][3] -
                a[3][1] * a[2][2] * a[1][3];

            inv[4] = -a[1][0] * a[2][2] * a[3][3] +
                a[1][0] * a[3][2] * a[2][3] +
                a[2][0] * a[1][2] * a[3][3] -
                a[2][0] * a[3][2] * a[1][3] -
                a[3][0] * a[1][2] * a[2][3] +
                a[3][0] * a[2][2] * a[1][3];

            inv[8] = a[1][0] * a[2][1] * a[3][3] -
                a[1][0] * a[3][1] * a[2][3] -
                a[2][0] * a[1][1] * a[3][3] +
                a[2][0] * a[3][1] * a[1][3] +
                a[3][0] * a[1][1] * a[2][3] -
                a[3][0] * a[2][1] * a[1][3];

            inv[12] = -a[1][0] * a[2][1] * a[3][2] +
                a[1][0] * a[3][1] * a[2][2] +
                a[2][0] * a[1][1] * a[3][2] -
                a[2][0] * a[3][1] * a[1][2] -
                a[3][0] * a[1][1] * a[2][2] +
                a[3][0] * a[2][1] * a[1][2];

            inv[1] = -a[0][1] * a[2][2] * a[3][3] +
                a[0][1] * a[3][2] * a[2][3] +
                a[2][1] * a[0][2] * a[3][3] -
                a[2][1] * a[3][2] * a[0][3] -
                a[3][1] * a[0][2] * a[2][3] +
                a[3][1] * a[2][2] * a[0][3];

            inv[5] = a[0][0] * a[2][2] * a[3][3] -
                a[0][0] * a[3][2] * a[2][3] -
                a[2][0] * a[0][2] * a[3][3] +
                a[2][0] * a[3][2] * a[0][3] +
                a[3][0] * a[0][2] * a[2][3] -
                a[3][0] * a[2][2] * a[0][3];

            inv[9] = -a[0][0] * a[2][1] * a[3][3] +
                a[0][0] * a[3][1] * a[2][3] +
                a[2][0] * a[0][1] * a[3][3] -
                a[2][0] * a[3][1] * a[0][3] -
                a[3][0] * a[0][1] * a[2][3] +
                a[3][0] * a[2][1] * a[0][3];

            inv[13] = a[0][0] * a[2][1] * a[3][2] -
                a[0][0] * a[3][1] * a[2][2] -
                a[2][0] * a[0][1] * a[3][2] +
                a[2][0] * a[3][1] * a[0][2] +
                a[3][0] * a[0][1] * a[2][2] -
                a[3][0] * a[2][1] * a[0][2];

            inv[2] = a[0][1] * a[1][2] * a[3][3] -
                a[0][1] * a[3][2] * a[1][3] -
                a[1][1] * a[0][2] * a[3][3] +
                a[1][1] * a[3][2] * a[0][3] +
                a[3][1] * a[0][2] * a[1][3] -
                a[3][1] * a[1][2] * a[0][3];

            inv[6] = -a[0][0] * a[1][2] * a[3][3] +
                a[0][0] * a[3][2] * a[1][3] +
                a[1][0] * a[0][2] * a[3][3] -
                a[1][0] * a[3][2] * a[0][3] -
                a[3][0] * a[0][2] * a[1][3] +
                a[3][0] * a[1][2] * a[0][3];

            inv[10] = a[0][0] * a[1][1] * a[3][3] -
                a[0][0] * a[3][1] * a[1][3] -
                a[1][0] * a[0][1] * a[3][3] +
                a[1][0] * a[3][1] * a[0][3] +
                a[3][0] * a[0][1] * a[1][3] -
                a[3][0] * a[1][1] * a[0][3];

            inv[14] = -a[0][0] * a[1][1] * a[3][2] +
                a[0][0] * a[3][1] * a[1][2] +
                a[1][0] * a[0][1] * a[3][2] -
                a[1][0] * a[3][1] * a[0][2] -
                a[3][0] * a[0][1] * a[1][2] +
                a[3][0] * a[1][1] * a[0][2];

            inv[3] = -a[0][1] * a[1][2] * a[2][3] +
                a[0][1] * a[2][2] * a[1][3] +
                a[1][1] * a[0][2] * a[2][3] -
                a[1][1] * a[2][2] * a[0][3] -
                a[2][1] * a[0][2] * a[1][3] +
                a[2][1] * a[1][2] * a[0][3];

            inv[7] = a[0][0] * a[1][2] * a[2][3] -
                a[0][0] * a[2][2] * a[1][3] -
                a[1][0] * a[0][2] * a[2][3] +
                a[1][0] * a[2][2] * a[0][3] +
                a[2][0] * a[0][2] * a[1][3] -
                a[2][0] * a[1][2] * a[0][3];

            inv[11] = -a[0][0] * a[1][1] * a[2][3] +
                a[0][0] * a[2][1] * a[1][3] +
                a[1][0] * a[0][1] * a[2][3] -
                a[1][0] * a[2][1] * a[0][3] -
                a[2][0] * a[0][1] * a[1][3] +
                a[2][0] * a[1][1] * a[0][3];

            inv[15] = a[0][0] * a[1][1] * a[2][2] -
                a[0][0] * a[2][1] * a[1][2] -
                a[1][0] * a[0][1] * a[2][2] +
                a[1][0] * a[2][1] * a[0][2] +
                a[2][0] * a[0][1] * a[1][2] -
                a[2][0] * a[1][1] * a[0][2];

            var det: T = a[0][0] * inv[0] + a[0][1] * inv[4] + a[0][2] * inv[8] + a[0][3] * inv[12];
            if (det == 0) return null;

            det = 1 / det;

            var out: Self = undefined;
            for (inv, 0..inv.len) |val, i| {
                out.cols[i % 4][i / 4] = val * det;
            }
            return out;
        }
    };
}

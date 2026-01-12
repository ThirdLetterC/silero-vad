const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const ort_include = b.option([]const u8, "ort-include", "Path to ONNX Runtime headers (e.g., /usr/include)");
    const ort_lib = b.option([]const u8, "ort-lib", "Path to ONNX Runtime libraries (e.g., /usr/lib)");

    const local_ort_root = "onnxruntime-linux-x64-1.18.0";
    const local_ort_include = b.pathJoin(&.{ local_ort_root, "include" });
    const local_ort_lib = b.pathJoin(&.{ local_ort_root, "lib" });

    const cwd = std.fs.cwd();
    const have_local_include = dirExists(cwd, local_ort_include);
    const have_local_lib = dirExists(cwd, local_ort_lib);

    const exe = b.addExecutable(.{
        .name = "silero_vad",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    exe.addCSourceFiles(.{
        .files = &.{
            "src/main.c",
            "src/silero_vad.c",
            "src/wav.c",
        },
        .flags = &.{
            "-std=c23",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
        },
    });

    exe.addIncludePath(.{ .src_path = .{ .owner = b, .sub_path = "src/include" } });
    if (ort_include) |inc| {
        exe.addIncludePath(.{ .cwd_relative = inc });
    } else if (have_local_include) {
        exe.addIncludePath(.{ .cwd_relative = local_ort_include });
    }

    if (ort_lib) |lib_path| {
        exe.addLibraryPath(.{ .cwd_relative = lib_path });
    } else if (have_local_lib) {
        exe.addLibraryPath(.{ .cwd_relative = local_ort_lib });
    }

    exe.linkLibC();
    exe.linkSystemLibrary("onnxruntime");

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Build and run the VAD demo");
    run_step.dependOn(&run_cmd.step);
}

fn dirExists(fs: std.fs.Dir, path: []const u8) bool {
    const result = fs.statFile(path) catch return false;
    return result.kind == .directory;
}

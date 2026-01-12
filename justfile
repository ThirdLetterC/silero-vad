set shell := ["bash", "-uc"]

build:
    zig build

run *args:
    zig build run -- {{args}}

fmt:
    zig fmt build.zig
    clang-format -i src/*.c src/include/*.h

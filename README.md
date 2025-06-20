# z3dm

**Zig 3D Math Library**

`z3dm` is a straightforward 3D math library written in Zig.
It’s designed to be simple and reliable — not overly complex or clever, but effective and easy to use.

This project is new and evolving.
Contributions, feedback, and suggestions are warmly welcome!

## Installation

`zig fetch --save git+https://github.com/HaraldWik/z3dm`

```rust
const z3dmn_dep = b.dependency("z3dm", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.addImport("z3dm", z3dmn_dep.module("z3dm"));
```

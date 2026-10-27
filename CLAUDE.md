# RENDER — working notes

A backend-neutral scene renderer interface (`render::`, `<render/Render.h>`, target `render::api`) and its Diligent
Engine backend (`render::diligent`, `<render/Diligent.h>`, target `render::diligent`). Moved out of the Scene Editor;
the Scene Editor and Behavior Kit fetch it by commit.

## Rules
- No Diligent type crosses `<render/Diligent.h>` (pimpl): consumers need neither Diligent's include paths nor its macros.
- The interface stays free of every consumer's types (no Skeleton / Pose of an editor): bones are `DrawBones` over
  joint positions and parent indices.
- The `ISceneRenderer` contract is GL-style (Z up, right-handed, CCW front faces, NDC z in -1..1); conversions live in
  `src/MathConv.h` only.
- Never set `CMAKE_MSVC_RUNTIME_LIBRARY`: it inherits the consumer's.
- Keep the Diligent pins in `CMakeLists.txt` in step with any consumer that builds its own Diligent.
- Tests: `ctest -LE gpu` runs the CPU suites; run `render-gpu-tests --backend vulkan` and `--backend d3d12` after any
  backend change.

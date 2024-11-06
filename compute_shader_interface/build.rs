use std::{env, path::PathBuf};
use std::error::Error;

use spirv_builder::SpirvBuilder;

fn main() -> Result<(), Box<dyn Error>> {

    let target_os = std::env::var("CARGO_CFG_TARGET_OS")?;
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH")?;
    println!("{}", target_os);
    println!("{}",target_arch);


    // Ensure OUT_DIR and PROFILE are set
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR is not set");
    let profile = env::var("PROFILE").expect("PROFILE is not set");

    // Set environment variables for spirv_builder
    env::set_var("OUT_DIR", out_dir);
    env::set_var("PROFILE", profile);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_OS");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    // While OUT_DIR is set for both build.rs and compiling the crate, PROFILE is only set in
    // build.rs. So, export it to crate compilation as well.
    let profile = env::var("PROFILE").unwrap();
    println!("cargo:rustc-env=PROFILE={profile}");
    // if target_os != "android" && target_arch != "wasm32" {
    //     return Ok(());
    // }
   // let shader_path = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    //println!("{:?}", shader_path);

    let mut builder_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
   //let path_to_crate = builder_dir.join(path_to_crate);
     builder_dir = builder_dir.parent().unwrap().join("compute_shader");
    println!("{:?}", builder_dir);

    let result = SpirvBuilder::new(builder_dir, "spirv-unknown-spv1.3")
    //.capability(spirv_builder::Capability::AtomicFloat32AddEXT)
    //.extension("SPV_EXT_shader_atomic_float_add")
    .build()?;

    // if codegen_names {
    //     let out_dir = env::var_os("OUT_DIR").unwrap();
    //     let dest_path = Path::new(&out_dir).join("entry_points.rs");
    //     fs::create_dir_all(&out_dir).unwrap();
    //     fs::write(dest_path, result.codegen_entry_point_strings()).unwrap();
    // }
    println!("{:?}", result);




    Ok(())
    // let mut dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    // // Strip `$profile/build/*/out`.
    // let ok = dir.ends_with("out")
    //     && dir.pop()
    //     && dir.pop()
    //     && dir.ends_with("build")
    //     && dir.pop()
    //     && dir.ends_with(profile)
    //     && dir.pop();
    // assert!(ok);
    // // NOTE(eddyb) this needs to be distinct from the `--target-dir` value that
    // // `spirv-builder` generates in a similar way from `$OUT_DIR` and `$PROFILE`,
    // // otherwise repeated `cargo build`s will cause build script reruns and the
    // // rebuilding of `rustc_codegen_spirv` (likely due to common proc macro deps).
    // let dir = dir.join("compute-shader-interface-builder");
    // let status = std::process::Command::new("cargo")
    //     .args([
    //         "run",
    //         "--release",
    //         "-p",
    //         "compute-shader-interface-builder",
    //         "--target-dir",
    //     ])
    //     .arg(dir)
    //     .env_remove("CARGO_ENCODED_RUSTFLAGS")
    //     .stderr(std::process::Stdio::inherit())
    //     .stdout(std::process::Stdio::inherit())
    //     .status()?;
    // if !status.success() {
    //     if let Some(code) = status.code() {
    //         std::process::exit(code);
    //     } else {
    //         std::process::exit(1);
    //     }
    // }
}


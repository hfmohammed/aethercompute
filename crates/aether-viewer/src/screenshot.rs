use std::path::Path;

/// Write tightly packed RGBA8 pixels as a PNG.
pub fn save_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> anyhow::Result<()> {
    anyhow::ensure!(
        rgba.len() == (width * height * 4) as usize,
        "pixel buffer size mismatch"
    );
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut enc = png::Encoder::new(file, width, height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(rgba)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_valid_png() {
        let p = std::env::temp_dir().join(format!("aether-shot-{}.png", std::process::id()));
        save_png(&p, 2, 2, &[255; 16]).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn rejects_wrong_size() {
        assert!(save_png(Path::new("/nonexistent/x.png"), 2, 2, &[0; 3]).is_err());
    }
}

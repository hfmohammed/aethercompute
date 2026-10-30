use std::process::Command;

use serde::{Deserialize, Serialize};

/// Hardware/software environment of a run (benchmark methodology item 1).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MachineInfo {
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub chip: String,
    pub model: String,
    pub ram_bytes: u64,
    pub logical_cpus: usize,
    pub power_source: String,
    pub rustc: String,
    pub aether_version: String,
    pub build_profile: String,
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

impl MachineInfo {
    pub fn detect() -> Self {
        let mac = cfg!(target_os = "macos");
        let sysctl = |k: &str| if mac { run("sysctl", &["-n", k]) } else { None };
        let power_source = if mac {
            run("pmset", &["-g", "batt"])
                .and_then(|s| {
                    s.lines().next().map(|l| {
                        l.replace("Now drawing from", "")
                            .trim()
                            .trim_matches('\'')
                            .to_string()
                    })
                })
                .unwrap_or_else(|| "unknown".into())
        } else {
            "unknown".into()
        };
        Self {
            os: std::env::consts::OS.to_string(),
            os_version: if mac {
                run("sw_vers", &["-productVersion"]).unwrap_or_default()
            } else {
                run("uname", &["-r"]).unwrap_or_default()
            },
            arch: std::env::consts::ARCH.to_string(),
            chip: sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "unknown".into()),
            model: sysctl("hw.model").unwrap_or_else(|| "unknown".into()),
            ram_bytes: sysctl("hw.memsize")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0),
            logical_cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            power_source,
            rustc: run("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
            aether_version: env!("CARGO_PKG_VERSION").to_string(),
            build_profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .into(),
        }
    }
}

/// (commit sha, working tree dirty). Falls back to `GIT_SHA` or "unknown".
pub fn git_state() -> (String, bool) {
    let sha = run("git", &["rev-parse", "HEAD"])
        .or_else(|| std::env::var("GIT_SHA").ok())
        .unwrap_or_else(|| "unknown".into());
    let dirty = run("git", &["status", "--porcelain", "--untracked-files=no"]).is_some();
    (sha, dirty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_fills_basics() {
        let m = MachineInfo::detect();
        assert!(!m.os.is_empty() && !m.arch.is_empty());
        assert!(m.logical_cpus >= 1);
        assert!(m.rustc.contains("rustc") || m.rustc == "unknown");
    }

    #[test]
    fn git_state_does_not_panic() {
        let (sha, _) = git_state();
        assert!(!sha.is_empty());
    }
}

# Publishing checklist

Run these steps after `gh auth login` (or with `GH_TOKEN` set).

## 1. Create GitHub repository and push

```powershell
cd "C:\Users\jacob\School Stuff\Research\qualiphide_fir_hp_inference"
gh repo create WashU-Astroparticle-Lab/qualiphide_fir_hp_inference --public --source=. --remote=origin --push --description "Profile-likelihood inference for QUALIPHIDE FIR hidden-photon search"
```

If the repository already exists:

```powershell
git remote add origin https://github.com/WashU-Astroparticle-Lab/qualiphide_fir_hp_inference.git
git push -u origin main
```

## 2. Create release v0.1.0

```powershell
gh release create v0.1.0 --title "v0.1.0" --notes "Initial public release for QUALIPHIDE FIR hidden-photon profile-likelihood inference."
```

## 3. Enable Zenodo

1. Log in at https://zenodo.org and link your GitHub account.
2. Go to **Account settings → GitHub** and click **Sync now**.
3. Enable **qualiphide_fir_hp_inference**.
4. After the v0.1.0 release is ingested, copy the version DOI.
5. Replace `10.5281/zenodo.XXXXXXX` in `README.md` and `docs/paper_citation.bib`.
6. Commit and push the DOI update.

## 4. Paper (Overleaf)

1. Sync the Overleaf project locally.
2. Add the entry from `docs/paper_citation.bib` (with the real DOI).
3. Replace `FIXME` with `\citemethods{qualiphide_fir_hp_inference}`.

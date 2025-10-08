## MOVER: Mobility Modeling with User Variability and Enriched Representations

This is the Repo for our submission to [**ACM SIGSPATIAL Cup 2025: Human Mobility Prediction Challenge**](https://sigspatial2025.sigspatial.org/giscup/index.html)

Our method **Mover** combines trip-aware segmentation, enriched location profiles, and cluster-based user adaptation.

### 1) Data Settings
Update dataset [ROOT] path in: ./datasets/builtin.py

- Download **Trip Detection** [CSV files](https://drive.google.com/drive/folders/1RBAntvHz2HuaBwPrIqYZUkFGJEu20esY?usp=drive_link) (replace the original compeition data)

Place Under:

```text
[ROOT]
```

- Download **Location Profile** [CSV files](https://drive.google.com/drive/folders/1vmEBKMDNnczJGwoqSHXPwjKv6KGBjK0R?usp=drive_link)

Place Under:

```text
./datasets/location_profile/v2/
```

- Download **User Cluster** [CSV files](https://drive.google.com/drive/folders/1WlNiAxh2gWIffANecfQztpYh4MpVo5l0)

Place Under:

```text
./datasets/user_cluster/
```

### 2) Download Model Weights
- **Model Weights:** [Google Drive – Weights](https://drive.google.com/drive/folders/1kObdlzuYdBW8ZUpiPOsePWP49TJbNC3j?usp=drive_link)
Place under:

```text
./_runs/
```

### 3) Run Inference
bash ./_bash/inference.sh


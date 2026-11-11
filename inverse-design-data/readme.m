Datasets for the microstructure and acoustic structure inverse-design case studies.

```text
data/
├── microstructure/
│   ├── mnist_img_train.txt
│   ├── mnist_img_test.txt
│   ├── summary_psi_train_all.txt
│   └── summary_psi_test_all.txt
└── acoustic/
    └── band_info_2D.csv
```

- **Microstructure:** 60,000 training and 10,000 test samples. Each sample contains a 28 × 28 image and 13 energy-response values. Image and response files are paired row by row.
- **Acoustic structure:** 500 structural designs, each with 15 modes and 61 wavevector sampling points. The CSV contains geometry parameters and corresponding frequencies.

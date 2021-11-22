# CFEN-ViT-Dehazing

Complementary Feature Enhanced Network with Vision Transformer for Image Dehazing



# Test Your Datasets
1. Download the pretained models: 
2. Unzip them into the /checkpoints/xxx/;
3. The test images should be put in \[your test data root\]/**hazy**/;
4. Run the following commands:
    1). **Homogeneous dehazing** 
    _(RESIDE-SOTS Dataset)_: 
    > python test.py --dataroot \[Your testing data root\] --name iid_hlgvit_crs_gd4_cfs_v3_reside --n_feats 24 --hidden_dim_ratio 4 --sb --out_all --which_epoch 32
    _(O-HAZE Dataset)_:
    > python test.py --dataroot \[Your testing data root\] --name iid_hlgvit_crs_gd4_cfs_v3_ohaze --n_feats 24 --hidden_dim_ratio 4 --sb --out_all --which_epoch 20
    2). **Non-homogeneous dehazing** 
    _(NH-HAZE)_:
    > python test.py --dataroot \[Your testing data root\] --name iid_hlgvit_crs_gd4_cfs_v3_nhhaze --n_feats 24 --hidden_dim_ratio 4 --sb --out_all --which_epoch 20
    3). **Nighttime dehazing**
    > python test.py --dataroot \[Your testing data root\] --name iid_hlgvit_crs_gd4_cfs_v3_daytime_realworld --n_feats 24 --hidden_dim_ratio 2 --sb --out_all 
    4). **Real_world dehazing**
    > python test.py --dataroot \[Your testing data root\] --name iid_hlgvit_crs_gd4_cfs_v3_daytime_realworld --n_feats 24 --hidden_dim_ratio 2 --sb --out_all 
    

    
# Results
![Real-world Dehazing 0005_real_B](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0005_real_B.png)
![Real-world Dehazing 0005_fake_A](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0005_fake_A.png)

![Real-world Dehazing 0061_real_B](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0061_real_B.png)
![Real-world Dehazing 0061_fake_A](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0061_fake_A.png)

![Real-world Dehazing 0005_real_B](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0005_real_B.png)
![Real-world Dehazing 0005_fake_A](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0005_fake_A.png)

![Real-world Dehazing 0085_real_B](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0085_real_B.png)
![Real-world Dehazing 0085_fake_A](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0085_fake_A.png)

![Real-world Dehazing 0128_real_B](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0128_real_B.png)
![Real-world Dehazing 0128_fake_A](https://github.com/phoenixtreesky7/CFEN-ViT-Dehazing/blob/main/new_real_hazy_0128_fake_A.png)



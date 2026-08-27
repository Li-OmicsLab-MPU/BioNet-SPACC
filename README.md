# BioNet-SPACC: Biologically Informed Network-Regularized Sparse Consensus Clustering

This repository contains the **reference** implementation of the **BioNet-SPACC** analytical pipeline.

This code supports the findings **reported** in our manuscript:
**"Biologically Informed Network-Regularized Sparse Consensus Clustering (BioNet-SPACC) Identifies A Continuous Plasma Proteomic Axis In Major Depressive Disorder"**


## Repository Structure

```text
BioNet-Spacc/
├── config/
│   └── config.yaml                 
├── data/                           
├── results/                        
├── src/
│   ├── utils/
│   │   └── network.py              
│   ├── BioNet_SPACC.py             
│   ├── stage0_data_pipeline.py    
│   ├── stage1_ppi_network.py       
│   ├── stage2_grid_search.py       
│   ├── stage3_final_clustering.py 
│   ├── stage4_validation.py       
│   └── stage5_classification.py   
├── requirements.txt
└── README.md

import os
import yaml
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from missingpy import MissForest

from utils.preprocessing import (
    filter_by_missing_rate,
    ClinicalImputer,
    LimmaCorrector,
)

class DataPipeline:
    def __init__(self, config_path='../config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.output_dir = self.config['data']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        self.random_seed = self.config.get('split', {}).get('random_state', 42)

    def load_and_align_data(self):
        df_protein = pd.read_csv(self.config['data']['protein_file'], index_col=0)
        df_cov = pd.read_csv(self.config['data']['baseline_file'], index_col=0)
        common_ids = df_protein.index.intersection(df_cov.index).sort_values()
        self.df_protein = df_protein.loc[common_ids].sort_index(axis=1)
        self.df_cov = df_cov.loc[common_ids]

    def create_stratification_key(self):
        age_binned = pd.cut(self.df_cov['Age_at_blood_draw'], bins=[0, 50, 60, 150], labels=['<50', '50-60', '>60'], right=False)
        strat_key = age_binned.astype(str) + "_" + self.df_cov['Sex'].astype(str) + "_" + self.df_cov['Comorbidity_Score'].astype(str)

        value_counts = strat_key.value_counts()
        rare_classes = value_counts[value_counts < 3].index
        if len(rare_classes) > 0:
            strat_key = strat_key.replace(rare_classes, 'Other_Rare')
        return strat_key

    def split_data(self):
        strat_key = self.create_stratification_key()
        test_size = self.config['split'].get('test_size', 0.20)

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=self.random_seed)
        train_idx, test_idx = next(splitter.split(self.df_protein, strat_key))
        self.prot_train = self.df_protein.iloc[train_idx]
        self.prot_test = self.df_protein.iloc[test_idx]
        self.cov_train = self.df_cov.iloc[train_idx]
        self.cov_test = self.df_cov.iloc[test_idx]
        self.cov_train.to_csv(f"{self.output_dir}/0.Cov_train_raw.csv")
        self.cov_test.to_csv(f"{self.output_dir}/0.Cov_test_raw.csv")
        self.prot_train.to_csv(f"{self.output_dir}/0.X_train_raw.csv")
        self.prot_test.to_csv(f"{self.output_dir}/0.X_test_raw.csv")

    def run_preprocessing_pipeline(self):
        clin_imputer = ClinicalImputer()
        cov_train_clean = clin_imputer.fit_transform(self.cov_train)
        cov_test_clean = clin_imputer.transform(self.cov_test)
        threshold = self.config['qc']['protein_missing_threshold']
        prot_train_qc, prot_test_qc, kept_proteins = filter_by_missing_rate(
            self.prot_train, self.prot_test, threshold
        )

        new_columns = [
            col[1:].replace('_i0', '').upper() if col.startswith('p') else col.replace('_i0', '').upper()
            for col in prot_train_qc.columns
        ]
        prot_train_qc.columns = new_columns
        prot_test_qc.columns = new_columns
        prot_train_qc = prot_train_qc.sort_index(axis=1)
        prot_test_qc = prot_test_qc.sort_index(axis=1)

        self.kept_proteins = list(prot_train_qc.columns)
        import pandas as pd
        pd.Series(self.kept_proteins).to_csv(
            f"{self.output_dir}/0.proteins_after_qc.csv",
            index=False, header=False
        )
        train_imp_path = f"{self.output_dir}/prot_train_imputed.csv"
        test_imp_path = f"{self.output_dir}/prot_test_imputed.csv"

        if os.path.exists(train_imp_path) and os.path.exists(test_imp_path):
            prot_train_imp = pd.read_csv(train_imp_path, index_col=0)
            prot_test_imp = pd.read_csv(test_imp_path, index_col=0)
        else:
            mf_imputer = MissForest(max_iter=10, n_jobs=-1, random_state=self.random_seed, n_estimators=50,
                                    max_features='sqrt', min_samples_leaf=2)

            X_train_imputed = mf_imputer.fit_transform(prot_train_qc.values)
            prot_train_imp = pd.DataFrame(X_train_imputed, index=prot_train_qc.index, columns=prot_train_qc.columns)
            X_test_imputed = mf_imputer.transform(prot_test_qc.values)
            prot_test_imp = pd.DataFrame(X_test_imputed, index=prot_test_qc.index, columns=prot_test_qc.columns)
            prot_train_imp.to_csv(train_imp_path)
            prot_test_imp.to_csv(test_imp_path)

        fixed_vars = ['Age_at_blood_draw', 'Sex', 'BMI', 'Duration_Years', 'Comorbidity_Score', 'Took_Antidepressant', 'Took_NSAIDs', 'Took_Statins']
        actual_vars = [v for v in fixed_vars if v in cov_train_clean.columns]

        limma = LimmaCorrector(formula_vars=actual_vars)
        prot_train_res = limma.fit_transform(prot_train_imp, cov_train_clean)
        prot_test_res = limma.transform(prot_test_imp, cov_test_clean)

        scaler = StandardScaler()
        prot_train_final = pd.DataFrame(scaler.fit_transform(prot_train_res), index=prot_train_res.index, columns=prot_train_res.columns)
        prot_test_final = pd.DataFrame(scaler.transform(prot_test_res), index=prot_test_res.index, columns=prot_test_res.columns)

        prot_train_final.to_csv(f"{self.output_dir}/1.X_train_final.csv")
        prot_test_final.to_csv(f"{self.output_dir}/1.X_test_final.csv")
        cov_train_clean.to_csv(f"{self.output_dir}/1.Cov_train_clean.csv")
        cov_test_clean.to_csv(f"{self.output_dir}/1.Cov_test_clean.csv")

        import joblib
        model_dir = f"{self.output_dir}/models_cache"
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(clin_imputer, f"{model_dir}/clin_imputer.pkl")
        joblib.dump(limma, f"{model_dir}/limma_corrector.pkl")
        joblib.dump(scaler, f"{model_dir}/scaler_X.pkl")

        train_protein_means = prot_train_imp.mean()
        train_protein_means.to_csv(f"{model_dir}/train_protein_means.csv")

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.load_and_align_data()
    pipeline.split_data()
    pipeline.run_preprocessing_pipeline()
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import yaml
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import precision_score, recall_score
from sklearn.utils import resample
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class ClassificationValidation:
    def __init__(self,
                 config_path=ROOT_DIR/'config'/'config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.output_dir = ROOT_DIR / self.config['data']['output_dir']
        self.final_cluster_dir = ROOT_DIR / self.config['data']['final_cluster_dir']
        self.results_dir4b = ROOT_DIR / self.config['data']['results_dir4b']

    def load_data(self):
        core_proteins = pd.read_csv(f"{self.final_cluster_dir}/Core_Proteins.csv")
        self.X_train_raw = pd.read_csv(f"{self.output_dir}/1.X_train_final.csv", index_col=0)[core_proteins['Protein']]
        self.y_train = pd.read_csv(f"{self.final_cluster_dir}/Y_Labels_Discovery.csv", index_col=0)['Subtype']
        self.X_test_raw = pd.read_csv(f"{self.output_dir}/1.X_test_final.csv", index_col=0)[core_proteins['Protein']]
        self.y_test = pd.read_csv(f"{self.final_cluster_dir}/Y_Labels_Validation.csv", index_col=0)['Subtype']
        print(f"[Training set] {self.X_train_raw.shape}")
        print(f"[Testing set] {self.X_test_raw.shape}")

    def cross_validation(self):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = {'XGBoost': [], 'RandomForest': []}
        X = self.X_train_raw.values
        y = self.y_train.values
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            weights = compute_sample_weight(class_weight='balanced', y=y_tr)
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            xgb = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric='mlogloss')
            xgb.fit(X_tr_scaled, y_tr, sample_weight=weights)
            pred_xgb = xgb.predict(X_val_scaled)
            cv_results['XGBoost'].append(f1_score(y_val, pred_xgb, average='weighted'))
            rf = RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42)
            rf.fit(X_tr, y_tr)
            pred_rf = rf.predict(X_val)
            cv_results['RandomForest'].append(f1_score(y_val, pred_rf, average='weighted'))
        for model, scores in cv_results.items():
            print(f"{model}: F1={np.mean(scores):.3f}±{np.std(scores):.3f}")

        return cv_results

    def external_test(self):
        rf_final = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, class_weight='balanced', random_state=42))
        ])
        rf_final.fit(self.X_train_raw, self.y_train)
        pred = rf_final.predict(self.X_test_raw)
        pred_proba_full = rf_final.predict_proba(self.X_test_raw)
        num_classes = pred_proba_full.shape[1]
        from sklearn.metrics import roc_auc_score
        n_iterations = 1000
        stats_boot = {'f1': [], 'precision': [], 'recall': [], 'acc': [], 'auc': []}
        y_true = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
        n_samples = len(y_true)

        for i in range(n_iterations):
            indices = resample(range(n_samples), replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = pred[indices]
            y_prob_boot = pred_proba_full[indices]
            stats_boot['f1'].append(f1_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            stats_boot['precision'].append(
                precision_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            stats_boot['recall'].append(recall_score(y_true_boot, y_pred_boot, average='weighted', zero_division=0))
            stats_boot['acc'].append(accuracy_score(y_true_boot, y_pred_boot))
            try:
                if num_classes == 2:
                    stats_boot['auc'].append(roc_auc_score(y_true_boot, y_prob_boot[:, 1]))
                else:
                    stats_boot['auc'].append(
                        roc_auc_score(y_true_boot, y_prob_boot, multi_class='ovr', average='weighted'))
            except ValueError:
                pass

        acc = accuracy_score(self.y_test, pred)
        f1 = f1_score(self.y_test, pred, average='weighted')
        prec = precision_score(self.y_test, pred, average='weighted')
        rec = recall_score(self.y_test, pred, average='weighted')
        if num_classes == 2:
            auc_score = roc_auc_score(self.y_test, pred_proba_full[:, 1])
        else:
            auc_score = roc_auc_score(self.y_test, pred_proba_full, multi_class='ovr', average='weighted')

        results = {}
        for metric, values in stats_boot.items():
            if len(values) > 0:
                results[f'{metric}_low'] = np.percentile(values, 2.5)
                results[f'{metric}_high'] = np.percentile(values, 97.5)

        print(f"Accuracy:  {acc:.3f} (95% CI: [{results['acc_low']:.3f}, {results['acc_high']:.3f}])")
        print(f"AUC:       {auc_score:.3f} (95% CI: [{results['auc_low']:.3f}, {results['auc_high']:.3f}])")
        print(f"Precision: {prec:.3f} (95% CI: [{results['precision_low']:.3f}, {results['precision_high']:.3f}])")
        print(f"Recall:    {rec:.3f} (95% CI: [{results['recall_low']:.3f}, {results['recall_high']:.3f}])")
        print(f"F1-Score:  {f1:.3f} (95% CI: [{results['f1_low']:.3f}, {results['f1_high']:.3f}])")

        print("\nClassification Report:")
        print(classification_report(self.y_test, pred))

        test_preds_dict = {
            'True_Subtype': self.y_test,
            'Predicted_Subtype': pred
        }
        for c in range(num_classes):
            test_preds_dict[f'Prob_Subtype_{c}'] = pred_proba_full[:, c]
        test_preds_df = pd.DataFrame(test_preds_dict, index=self.X_test_raw.index)
        out_pred_path = f"{self.results_dir4b}/Y_test_Predicted_85plex.csv"
        test_preds_df.to_csv(out_pred_path)

        return {
            'Accuracy': acc,
            'Acc_CI_Low': results['acc_low'],
            'Acc_CI_High': results['acc_high'],
            'AUC': auc_score, 'AUC_CI_Low': results['auc_low'], 'AUC_CI_High': results['auc_high'],
            'F1': f1,
            'F1_CI_Low': results['f1_low'],
            'F1_CI_High': results['f1_high'],
            'Precision': prec,
            'Prec_CI_Low': results['precision_low'],
            'Prec_CI_High': results['precision_high'],
            'Recall': rec,
            'Rec_CI_Low': results['recall_low'],
            'Rec_CI_High': results['recall_high']
        }

    def run(self):
        self.load_data()
        cv_res = self.cross_validation()
        test_res = self.external_test()
        summary = {
            'CV_XGBoost_F1': f"{np.mean(cv_res['XGBoost']):.3f}±{np.std(cv_res['XGBoost']):.3f}",
            'CV_RF_F1': f"{np.mean(cv_res['RandomForest']):.3f}±{np.std(cv_res['RandomForest']):.3f}",
            'Test_Accuracy': f"{test_res['Accuracy']:.3f} [{test_res['Acc_CI_Low']:.3f}-{test_res['Acc_CI_High']:.3f}]",
            'Test_F1': f"{test_res['F1']:.3f} [{test_res['F1_CI_Low']:.3f}-{test_res['F1_CI_High']:.3f}]"
        }
        pd.DataFrame([summary]).to_csv(f"{self.results_dir4b}/Classification_Summary.csv", index=False)

if __name__ == "__main__":
    stage4b = ClassificationValidation()
    stage4b.run()
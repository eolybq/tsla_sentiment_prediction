import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.tree import export_graphviz
from xgboost import plot_importance, to_graphviz
import graphviz

# TODO - mozna udelat funkci oddelene primo pro GD modely a pak jen evaluate_class a evaluate_regg ? vypada ta evaluace uplne stejne tak at neni duplicat kod
# - pak primo pro dt, rf, xgb? xgb trochu jinak ale idk asi nechat jak je

custom_colors = {
    'y_pred': '#1717c1',
    'y_test': '#4D4D4D',
    'Actual': '#4D4D4D',
    'Predicted': '#1717c1'
}


# GD LR
def evaluate_gd_lr(y_test, y_pred_test):
    # Metrics
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # naive forecast: y_{t-1}
    naive_pred = y_test.shift(1).dropna()
    naive_mse = mean_squared_error(y_test.iloc[1:], naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    naive_mae = mean_absolute_error(y_test.iloc[1:], naive_pred)

    test_mase = test_mae / naive_mae

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Real: {y_test.values[-1]}, Predikce: {y_pred_test[-1]}")

    # R^2
    sse_test = np.sum((y_test.values - y_pred_test) ** 2)
    sst_test = np.sum((y_test.values - y_test.mean()) ** 2)

    r2_test = 1 - sse_test / sst_test
    print(f"Test R^2: {r2_test:.4f}")

    metrics_df = pd.DataFrame({
        "Model": ["GD LinearRegression", "Naive"],
        "MSE": [test_mse, naive_mse],
        "RMSE": [test_rmse, naive_rmse],
        "MAE": [test_mae, naive_mae],
        "MASE": [test_mase, 1.0000],
        "R^2": [r2_test, ""]
    })
    metrics_df.to_excel("../plots_tabs/gd_lr_metrics.xlsx", index=False)


    # Plots
    y_test_df = pd.DataFrame({
        'log_return': y_test,
        'variable': 'y_test'
    }).reset_index()

    y_pred_df = pd.DataFrame({
        'log_return': y_pred_test,
        'variable': 'y_pred'
    }).reset_index()

    test_pred_df = pd.concat([y_test_df, y_pred_df], ignore_index=True, axis=0)

    sns.lineplot(data=test_pred_df, x="index", y='log_return', hue='variable', palette=custom_colors)
    plt.xlabel('')
    plt.ylabel('Log return')
    plt.title("Gradient Descent LinearRegression")
    plt.savefig("../plots_tabs/gd_lr.png", dpi=300, bbox_inches='tight')
    plt.show()



def evaluate_gd_logit(y_test, y_pred_class, y_pred_proba):
    acc = accuracy_score(y_test, y_pred_class)
    log_l = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics_df = pd.DataFrame({
        "Model": ["GD LogisticRegression"],
        "Accuracy": [acc],
        "Log Loss": [log_l],
        "ROC AUC": [roc_auc]
    })

    metrics_df.to_excel("../plots_tabs/gd_log_metrics.xlsx", index=False)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Log loss: {log_l:.6f}")
    print(f"Roc auc: {roc_auc:.6f}")

    print(classification_report(y_test, y_pred_class))

    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("GD Logit Confusion Matrix")
    plt.savefig("../plots_tabs/gd_log_conf_m.png", dpi=300, bbox_inches='tight')
    plt.show()


    df_plot = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_class
    })

    sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette=custom_colors)
    plt.title("Gradient Descent LogisticRegression")
    plt.xlabel("")
    plt.savefig("../plots_tabs/gd_log_count_p.png", dpi=300, bbox_inches='tight')
    plt.show()



def evaluate_dtc(y_test, y_pred_class, y_pred_proba, dtc, features_names):
    dot_data = export_graphviz(dtc, out_file=None,
                               feature_names=features_names,
                               class_names=['0','1'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("../plots_tabs/dtc", cleanup=True)
    graph.view()


    # Feature importance
    dtc_importances = pd.Series(dtc.feature_importances_, index=features_names)
    # jen použité feature
    dtc_importances_used = dtc_importances[dtc_importances > 0].sort_values(ascending=False)

    dtc_importances_used.plot(kind='bar', figsize=(10,10))
    plt.title("DTC Feature Importances")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.show()


    acc = accuracy_score(y_test, y_pred_class)
    log_l = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics_df = pd.DataFrame({
        "Model": ["DecisionTreeClassifier"],
        "Accuracy": [acc],
        "Log Loss": [log_l],
        "ROC AUC": [roc_auc]
    })

    metrics_df.to_excel("../plots_tabs/dtc_metrics.xlsx", index=False)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Log loss: {log_l:.6f}")
    print(f"Roc auc: {roc_auc:.6f}")

    print(classification_report(y_test, y_pred_class))

    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("DTC Confusion Matrix")
    plt.savefig("../plots_tabs/dtc_conf_m.png", dpi=300, bbox_inches='tight')
    plt.show()


    df_plot = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_class
    })

    sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette=custom_colors)
    plt.title("Decision Tree Classifier")
    plt.xlabel("")
    plt.savefig("../plots_tabs/dtc_count_p.png", dpi=300, bbox_inches='tight')
    plt.show()




def evaluate_dtr(y_test, y_pred, dtr, features_names):
    dot_data = export_graphviz(dtr, out_file=None,
                               feature_names=features_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("../plots_tabs/dtr", cleanup=True)
    graph.view()

    # Feature importance
    dtr_importances = pd.Series(dtr.feature_importances_, index=features_names)

    # jen použité feature
    dtr_importances_used = dtr_importances[dtr_importances > 0].sort_values(ascending=False)

    dtr_importances_used.plot(kind='bar', figsize=(10, 10))
    plt.title("DTR Feature Importances")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.show()


    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_pred)

    # naive forecast: y_{t-1}
    naive_pred = y_test[:-1]
    naive_mse = mean_squared_error(y_test[1:], naive_pred)
    naive_rmse = np.sqrt(naive_mse)
    naive_mae = mean_absolute_error(y_test[1:], naive_pred)

    test_mase = test_mae / naive_mae

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Real: {y_test[-1]}, Predikce: {y_pred[-1]}")

    # R^2
    sse_test = np.sum((y_test - y_pred) ** 2)
    sst_test = np.sum((y_test - y_test.mean()) ** 2)

    r2_test = 1 - sse_test / sst_test
    print(f"Test R^2: {r2_test:.4f}")

    metrics_df = pd.DataFrame({
        "Model": ["DecisionTreeRegressor", "Naive"],
        "MSE": [test_mse, naive_mse],
        "RMSE": [test_rmse, naive_rmse],
        "MAE": [test_mae, naive_mae],
        "MASE": [test_mase, 1.0000],
        "R^2": [r2_test, ""]
    })
    metrics_df.to_excel("../plots_tabs/dtr_metrics.xlsx", index=False)


    y_test_df = pd.DataFrame({
        'log_return': y_test,
        'variable': 'y_test'
    }).reset_index()

    y_pred_df = pd.DataFrame({
        'log_return': y_pred,
        'variable': 'y_pred'
    }).reset_index()

    test_pred_df = pd.concat([y_test_df, y_pred_df], ignore_index=True, axis=0)

    sns.lineplot(data=test_pred_df, x="index", y='log_return', hue='variable', palette=custom_colors)
    plt.xlabel('')
    plt.ylabel('Log return')
    plt.title("Decision Tree Regressor")
    plt.savefig("../plots_tabs/dtr_preds.png", dpi=300, bbox_inches='tight')
    plt.show()




def evaluate_rfc(y_test, y_pred_class, y_pred_proba, rfc, features_names):
    dot_data = export_graphviz(rfc.estimators_[0], out_file=None,
                               feature_names=features_names,
                               class_names=['0','1'],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("../plots_tabs/rfc_first_tree", cleanup=True)
    graph.view()


    # Feature importance
    rfc_importances = pd.Series(rfc.feature_importances_, index=features_names)

    # jen použité feature
    rfc_importances_used = rfc_importances[rfc_importances > 0].sort_values(ascending=False)

    rfc_importances_used.plot(kind='bar', figsize=(10,10))
    plt.title("RFC Feature Importances")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.show()

    acc = accuracy_score(y_test, y_pred_class)
    log_l = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics_df = pd.DataFrame({
        "Model": ["RandomForestClassifier"],
        "Accuracy": [acc],
        "Log Loss": [log_l],
        "ROC AUC": [roc_auc]
    })

    metrics_df.to_excel("../plots_tabs/rfc_metrics.xlsx", index=False)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Log loss: {log_l:.6f}")
    print(f"Roc auc: {roc_auc:.6f}")

    print(classification_report(y_test, y_pred_class))

    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("RFC Confusion Matrix")
    plt.savefig("../plots_tabs/rfc_conf_m.png", dpi=300, bbox_inches='tight')
    plt.show()


    df_plot = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_class
    })

    sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette=custom_colors)
    plt.title("Random Forest Classifier")
    plt.xlabel("")
    plt.savefig("../plots_tabs/rfc_count_p.png", dpi=300, bbox_inches='tight')
    plt.show()



# TODO Random Forest regressor



def evaluate_xgbc(y_test, y_pred_class, y_pred_proba, xgbc):
    graph = to_graphviz(xgbc, num_trees=0)
    graph.render("../plots_tabs/xgbc_first_tree.png")
    graph.view()


    plot_importance(xgbc, importance_type='gain')
    plt.show()

    # type weight
    plot_importance(xgbc, importance_type='weight')
    plt.show()


    acc = accuracy_score(y_test, y_pred_class)
    log_l = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics_df = pd.DataFrame({
        "Model": ["XGBoostClassifier"],
        "Accuracy": [acc],
        "Log Loss": [log_l],
        "ROC AUC": [roc_auc]
    })

    metrics_df.to_excel("../plots_tabs/xgbc_metrics.xlsx", index=False)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Log loss: {log_l:.6f}")
    print(f"Roc auc: {roc_auc:.6f}")

    print(classification_report(y_test, y_pred_class))

    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("XGB Classifier Confusion Matrix")
    plt.savefig("../plots_tabs/xgbc_conf_m.png", dpi=300, bbox_inches='tight')
    plt.show()


    df_plot = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_class
    })

    sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette=custom_colors)
    plt.title("XGBoost Classifier")
    plt.xlabel("")
    plt.savefig("../plots_tabs/xgbc_count_p.png", dpi=300, bbox_inches='tight')
    plt.show()



# TODO XGB regressor



# TODO LIGHTGBM regressor
# TODO LIGHTGBM classifier

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, log_loss, roc_auc_score, confusion_matrix, classification_report
from sklearn.tree import export_graphviz, DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import plot_importance, to_graphviz, XGBClassifier, XGBRegressor
import lightgbm as lgb
import graphviz



custom_colors = {
    'y_pred': '#1717c1',
    'y_test': '#4D4D4D',
    'Actual': '#4D4D4D',
    'Predicted': '#1717c1'
}

k_importances = 25


def evaluate_class(y_test, y_pred_class, y_pred_proba, model_name, model=None, features_names=None):
    if model:
        if isinstance(model, DecisionTreeClassifier):
            dot_data = export_graphviz(model, out_file=None,
                                       feature_names=features_names,
                                       class_names=['0', '1'],
                                       filled=True, rounded=True,
                                       special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(f"python/plots_tabs/{model_name}", cleanup=True)
            graph.view()

            # Feature importance
            model_importances = pd.Series(model.feature_importances_, index=features_names)
            # jen použité feature
            model_importances_used = model_importances[model_importances > 0].sort_values(ascending=False)

            model_importances_used.plot(kind='bar', figsize=(20, 10))
            plt.title(f"{model_name} Feature Importances")
            plt.ylabel("Importance")
            plt.xlabel("Feature")
            plt.show()

        elif isinstance(model, RandomForestClassifier):
            dot_data = export_graphviz(model.estimators_[0], out_file=None,
                                       feature_names=features_names,
                                       class_names=['0', '1'],
                                       filled=True, rounded=True,
                                       special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(f"python/plots_tabs/{model_name}_first_tree", cleanup=True)
            graph.view()

            # Feature importance
            model_importances = pd.Series(model.feature_importances_, index=features_names)
            # jen použité feature
            model_importances_used = model_importances[model_importances > 0].sort_values(ascending=False)
            top_k_importances_used = model_importances_used.iloc[:k_importances]

            top_k_importances_used.plot(kind='bar', figsize=(20, 10))
            plt.title(f"{model_name} Feature Importances")
            plt.ylabel("Importance")
            plt.xlabel("Feature")
            plt.show()

        elif isinstance(model, XGBClassifier):
            graph = to_graphviz(model, num_trees=0)
            graph.render(f"python/plots_tabs/{model_name}_first_tree.png")
            graph.view()

            plot_importance(model, importance_type='gain', max_num_features=k_importances)
            plt.title(f"{model_name} Feature Importance (gain)")
            plt.show()

            plot_importance(model, importance_type='weight', max_num_features=k_importances)
            plt.title(f"{model_name} Feature Importance (weight)")
            plt.show()

        elif isinstance(model, lgb.LGBMClassifier):
            ax = lgb.plot_tree(model, tree_index=0, figsize=(20, 10))
            plt.title(f"{model_name} First Tree")
            plt.savefig(f"python/plots_tabs/{model_name}_first_tree.png", dpi=300, bbox_inches='tight')
            plt.show()

            # feature importance
            model_importances = pd.Series(model.feature_importances_, index=features_names)

            # jen použité feature
            model_importances_used = model_importances[model_importances > 0].sort_values(ascending=False)
            top_k_importances_used = model_importances_used.iloc[:k_importances]

            top_k_importances_used.plot(kind='bar', figsize=(20, 10))
            plt.title(f"{model_name} Feature Importances")
            plt.ylabel("Importance")
            plt.xlabel("Feature")
            plt.show()

    print(classification_report(y_test, y_pred_class))

    report_dict = classification_report(y_test, y_pred_class, output_dict=True)

    precision_avg = report_dict['weighted avg']['precision']
    recall_avg = report_dict['weighted avg']['recall']
    f1_avg = report_dict['weighted avg']['f1-score']

    acc = accuracy_score(y_test, y_pred_class)
    log_l = log_loss(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics_df = pd.DataFrame({
        "Model": [model_name],
        "Accuracy": [acc],
        "Log Loss": [log_l],
        "ROC AUC": [roc_auc],
        "Precision (weighted)": [precision_avg],
        "Recall (weighted)": [recall_avg],
        "F1-score (weighted)": [f1_avg]
    })

    metrics_df.to_excel(f"python/plots_tabs/{model_name}_metrics.xlsx", index=False)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Log loss: {log_l:.6f}")
    print(f"Roc auc: {roc_auc:.6f}")


    cm = confusion_matrix(y_test, y_pred_class)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(f"python/plots_tabs/{model_name}_conf_m.png", dpi=300, bbox_inches='tight')
    plt.show()


    df_plot = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred_class
    })

    sns.countplot(data=pd.melt(df_plot), x="value", hue="variable", palette=custom_colors)
    plt.title(model_name)
    plt.xlabel("")
    plt.savefig(f"python/plots_tabs/{model_name}_count_p.png", dpi=300, bbox_inches='tight')
    plt.show()




def evaluate_regg(y_test, y_pred, model_name, model=None, features_names=None):
    if model:
        if isinstance(model, DecisionTreeRegressor):
            dot_data = export_graphviz(model, out_file=None,
                                       feature_names=features_names,
                                       filled=True, rounded=True,
                                       special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(f"python/plots_tabs/{model_name}", cleanup=True)
            graph.view()

            # Feature importance
            model_importances = pd.Series(model.feature_importances_, index=features_names)

            # jen použité feature
            model_importances_used = model_importances[model_importances > 0].sort_values(ascending=False)

            model_importances_used.plot(kind='bar', figsize=(20, 10))
            plt.title(f"{model_name} Feature Importances")
            plt.ylabel("Importance")
            plt.xlabel("Feature")
            plt.show()

        elif isinstance(model, RandomForestRegressor):
            dot_data = export_graphviz(model.estimators_[0], out_file=None,
                                       feature_names=features_names,
                                       filled=True, rounded=True,
                                       special_characters=True)
            graph = graphviz.Source(dot_data)
            graph.render(f"python/plots_tabs/{model_name}_first_tree", cleanup=True)
            graph.view()

            # Feature importance
            model_importances = pd.Series(model.feature_importances_, index=features_names)
            # jen použité feature
            model_importances_used = model_importances[model_importances > 0].sort_values(ascending=False)
            top_k_importances_used = model_importances_used.iloc[:k_importances]

            top_k_importances_used.plot(kind='bar', figsize=(20, 10))
            plt.title(f"{model_name} Feature Importances")
            plt.ylabel("Importance")
            plt.xlabel("Feature")
            plt.show()

        elif isinstance(model, XGBRegressor):
            graph = to_graphviz(model, num_trees=0)
            graph.render(f"python/plots_tabs/{model_name}_first_tree.png")
            graph.view()

            plot_importance(model, importance_type='gain', max_num_features=k_importances)
            plt.title(f"{model_name} Feature Importance (gain)")
            plt.show()

            plot_importance(model, importance_type='weight', max_num_features=k_importances)
            plt.title(f"{model_name} Feature Importance (weight)")
            plt.show()

        elif isinstance(model, lgb.LGBMRegressor):
            ax = lgb.plot_tree(model, tree_index=0, figsize=(20, 10))
            plt.title(f"{model_name} First Tree")
            plt.savefig(f"python/plots_tabs/{model_name}_first_tree.png", dpi=300, bbox_inches='tight')
            plt.show()

            # feature importance
            model_importances = pd.Series(model.feature_importances_, index=features_names)

            # jen použité feature
            model_importances_used = model_importances[model_importances > 0].sort_values(ascending=False)
            top_k_importances_used = model_importances_used.iloc[:k_importances]

            top_k_importances_used.plot(kind='bar', figsize=(20, 10))
            plt.title(f"{model_name} Feature Importances")
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
    # print(f"Real: {y_test[-1]}, Predikce: {y_pred[-1]}")

    # R^2
    sse_test = np.sum((y_test - y_pred) ** 2)
    sst_test = np.sum((y_test - y_test.mean()) ** 2)

    r2_test = 1 - sse_test / sst_test
    print(f"Test R^2: {r2_test:.4f}")


    # Median errors and Directional Accuracy -> neni tolik ovlivneno velkymi chybami - outliers
    mse_median = np.median((y_test - y_pred) ** 2)
    mae_median = np.median(np.abs(y_test - y_pred))

    delta_true = np.diff(y_test)
    delta_pred = np.diff(y_pred)
    d_accuracy = np.mean(np.sign(delta_true) == np.sign(delta_pred))

    naive_mse_median = np.median((y_test[1:] - naive_pred) ** 2)
    naive_mae_median = np.median(np.abs(y_test[1:] - naive_pred))



    metrics_df = pd.DataFrame({
        "Model": [model_name, "Naive"],
        "MSE": [test_mse, naive_mse],
        "RMSE": [test_rmse, naive_rmse],
        "MAE": [test_mae, naive_mae],
        "MASE": [test_mase, 1.0000],
        "R^2": [r2_test, ""],
        "MSE_median": [mse_median, naive_mse_median],
        "MAE_median": [mae_median, naive_mae_median],
        "Dir_accuracy": [d_accuracy, ""],
    })
    metrics_df.to_excel(f"python/plots_tabs/{model_name}_metrics.xlsx", index=False)


    # Plots
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
    plt.title(model_name)
    plt.savefig(f"python/plots_tabs/{model_name}_preds.png", dpi=300, bbox_inches='tight')
    plt.show()
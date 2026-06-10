import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
    brier_score_loss, mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# Cache data once so the UI is responsive
@st.cache_data(show_spinner=False)
def load_titanic_data():
    df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv')
    df = df.drop(columns=['adult_male', 'embarked', 'class', 'alive', 'alone'])
    return df


@st.cache_data(show_spinner=False)
def load_housing_data():
    candidate_paths = [
        Path('data/housing_price.csv'),
        Path('housing_price.csv'),
        Path('housing-price.csv'),
        Path('data/housing.csv'),
        Path('housing.csv'),
    ]
    csv_path = next((path for path in candidate_paths if path.exists()), None)
    if csv_path is not None:
        return pd.read_csv(csv_path)

    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df['Price'] = df['MedHouseVal'] * 100000
    df = df.drop(columns=['MedHouseVal'])
    return df


def preprocess_titanic(df):
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['survived', 'pclass', 'age', 'fare'])

    le_sex = LabelEncoder()
    le_who = LabelEncoder()
    le_deck = LabelEncoder()
    le_embark_town = LabelEncoder()

    df_clean['sex_encoded'] = le_sex.fit_transform(df_clean['sex'])
    df_clean['who_encoded'] = le_who.fit_transform(df_clean['who'])
    df_clean['deck_encoded'] = le_deck.fit_transform(df_clean['deck'].fillna('Unknown'))
    df_clean['embark_town_encoded'] = le_embark_town.fit_transform(df_clean['embark_town'].fillna('Unknown'))

    feature_names = ['pclass', 'sex_encoded', 'age', 'sibsp', 'parch', 'fare',
                     'who_encoded', 'deck_encoded', 'embark_town_encoded']
    X = df_clean[feature_names]
    y = df_clean['survived']
    return X, y, df_clean


def find_housing_target(df):
    possible_targets = [
        'Price', 'price', 'SalePrice', 'saleprice', 'Sale Price', 'sale price',
        'HousePrice', 'houseprice'
    ]
    for name in possible_targets:
        if name in df.columns:
            return name

    numeric_cols = [col for col in df.select_dtypes(include=['number']).columns]
    candidate_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in ['price', 'sale', 'value'])]
    return candidate_cols[0] if candidate_cols else numeric_cols[-1]


def preprocess_housing(df):
    df_clean = df.copy()
    target_col = find_housing_target(df_clean)
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    return X, y, df_clean, target_col


def get_classification_model(model_name):
    if model_name == 'Decision Tree':
        return DecisionTreeClassifier(random_state=42, max_depth=10)
    if model_name == 'Random Forest':
        return RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    if model_name == 'SVM':
        return SVC(probability=True, random_state=42, kernel='rbf')
    if model_name == 'Logistic Regression':
        return LogisticRegression(max_iter=1000, random_state=42)
    if model_name == 'Naive Bayes':
        return GaussianNB()
    if model_name == 'Neural Network':
        return MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
    raise ValueError(f'Unknown classification model: {model_name}')


def get_regression_model(model_name): 
    if model_name == 'Linear Regression':
        return LinearRegression()
    if model_name == 'Ridge Regression':
        return Ridge(alpha=1.0, random_state=42)
    if model_name == 'Lasso Regression':
        return Lasso(alpha=0.1, random_state=42)
    if model_name == 'SVM':
        return SVR(kernel='rbf', C=1.0, epsilon=0.2)
    if model_name == 'Decision Tree':
        return DecisionTreeRegressor(random_state=42, max_depth=8)
    if model_name == 'Random Forest':
        return RandomForestRegressor(n_estimators=100, random_state=42, max_depth=12)
    if model_name == 'Neural Network':
        return MLPRegressor(hidden_layer_sizes=(10,), max_iter=1, random_state=42)
    raise ValueError(f'Unknown regression model: {model_name}')


def get_housing_preprocessor(X):
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols),
    ])


def evaluate_classification_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    out = {}
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    def get_probs(m, X):
        if hasattr(m, 'predict_proba'):
            return m.predict_proba(X)[:, 1]
        if hasattr(m, 'decision_function'):
            scores = m.decision_function(X)
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return None

    y_train_prob = get_probs(model, X_train)
    y_test_prob = get_probs(model, X_test)

    out['train'] = {
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'Precision': precision_score(y_train, y_train_pred),
        'Recall': recall_score(y_train, y_train_pred),
        'F1 score': f1_score(y_train, y_train_pred),
        'Confusion matrix': confusion_matrix(y_train, y_train_pred),
    }
    out['test'] = {
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1 score': f1_score(y_test, y_test_pred),
        'Confusion matrix': confusion_matrix(y_test, y_test_pred),
        'Classification report': classification_report(y_test, y_test_pred, output_dict=False),
    }
    out['train']['y_prob'] = y_train_prob
    out['test']['y_prob'] = y_test_prob

    if y_test_prob is not None:
        try:
            out['test']['ROC AUC'] = roc_auc_score(y_test, y_test_prob)
        except Exception:
            out['test']['ROC AUC'] = None
        out['test']['Brier score'] = brier_score_loss(y_test, y_test_prob)

    return out


def evaluate_regression_model(model, preprocessor, X_train, X_test, y_train, y_test):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    return {
        'model': pipeline,
        'train': {
            'RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'MAE': mean_absolute_error(y_train, y_train_pred),
            'R2': r2_score(y_train, y_train_pred),
        },
        'test': {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'R2': r2_score(y_test, y_test_pred),
        },
        'y_pred': y_test_pred,
        'residuals': y_test - y_test_pred,
    }


def evaluate_housing_naive_bayes(preprocessor, X_train, X_test, y_train, y_test, n_bins=4):
    y_train_bins, bins = pd.qcut(y_train, q=n_bins, labels=False, retbins=True, duplicates='drop')
    y_test_bins = pd.cut(y_test, bins=bins, labels=False, include_lowest=True)
    y_test_bins = y_test_bins.fillna(len(bins) - 2).astype(int)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', GaussianNB())])
    pipeline.fit(X_train, y_train_bins)
    y_pred_bins = pipeline.predict(X_test)

    bin_medians = y_train.groupby(y_train_bins).median()
    approx_price = pd.Series(y_pred_bins).map(bin_medians).astype(float).values

    return {
        'model': pipeline,
        'train': {
            'Accuracy': accuracy_score(y_train_bins, pipeline.predict(X_train)),
            'Precision': precision_score(y_train_bins, pipeline.predict(X_train), average='weighted', zero_division=0),
            'Recall': recall_score(y_train_bins, pipeline.predict(X_train), average='weighted', zero_division=0),
            'F1 score': f1_score(y_train_bins, pipeline.predict(X_train), average='weighted', zero_division=0),
            'Confusion matrix': confusion_matrix(y_train_bins, pipeline.predict(X_train)),
        },
        'test': {
            'Accuracy': accuracy_score(y_test_bins, y_pred_bins),
            'Precision': precision_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0),
            'Recall': recall_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0),
            'F1 score': f1_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0),
            'Confusion matrix': confusion_matrix(y_test_bins, y_pred_bins),
            'Classification report': classification_report(y_test_bins, y_pred_bins, output_dict=False),
            'Approx RMSE': np.sqrt(mean_squared_error(y_test, approx_price)),
            'Approx MAE': mean_absolute_error(y_test, approx_price),
            'Bin medians': bin_medians,
        },
        'y_pred': approx_price,
        'residuals': y_test - approx_price,
        'y_test_bins': y_test_bins,
        'y_pred_bins': y_pred_bins,
        'bin_medians': bin_medians,
    }


def evaluate_housing_classifier(preprocessor, X_train, X_test, y_train, y_test, model, n_bins=4):
    # Bin continuous target into quantile buckets for classification
    y_train_bins, bins = pd.qcut(y_train, q=n_bins, labels=False, retbins=True, duplicates='drop')
    y_test_bins = pd.cut(y_test, bins=bins, labels=False, include_lowest=True)
    y_test_bins = y_test_bins.fillna(len(bins) - 2).astype(int)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    pipeline.fit(X_train, y_train_bins)
    y_pred_bins = pipeline.predict(X_test)

    # Map predicted bins to median price within each bin to approximate continuous prediction
    bin_medians = y_train.groupby(y_train_bins).median()
    approx_price = pd.Series(y_pred_bins).map(bin_medians).astype(float).values

    return {
        'model': pipeline,
        'train': {
            'Accuracy': accuracy_score(y_train_bins, pipeline.predict(X_train)),
            'Precision': precision_score(y_train_bins, pipeline.predict(X_train), average='weighted', zero_division=0),
            'Recall': recall_score(y_train_bins, pipeline.predict(X_train), average='weighted', zero_division=0),
            'F1 score': f1_score(y_train_bins, pipeline.predict(X_train), average='weighted', zero_division=0),
            'Confusion matrix': confusion_matrix(y_train_bins, pipeline.predict(X_train)),
        },
        'test': {
            'Accuracy': accuracy_score(y_test_bins, y_pred_bins),
            'Precision': precision_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0),
            'Recall': recall_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0),
            'F1 score': f1_score(y_test_bins, y_pred_bins, average='weighted', zero_division=0),
            'Confusion matrix': confusion_matrix(y_test_bins, y_pred_bins),
            'Classification report': classification_report(y_test_bins, y_pred_bins, output_dict=False),
            'Approx RMSE': np.sqrt(mean_squared_error(y_test, approx_price)),
            'Approx MAE': mean_absolute_error(y_test, approx_price),
            'Bin medians': bin_medians,
        },
        'y_pred': approx_price,
        'residuals': y_test - approx_price,
        'y_test_bins': y_test_bins,
        'y_pred_bins': y_pred_bins,
        'bin_medians': bin_medians,
    }


def format_confusion_matrix(cm):
    return pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])


def main():
    st.set_page_config(page_title='Model Comparison UI', layout='wide')
    st.title('Dataset and Model Comparison')

    with st.sidebar:
        st.header('Controls')
        dataset_name = st.selectbox('Choose dataset', ['Titanic', 'Housing Price'])
        if dataset_name == 'Titanic':
            model_choice = st.selectbox(
                'Choose model',
                ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression', 'Naive Bayes', 'Neural Network', 'Compare models']
            )
        else:
            model_choice = st.selectbox(
                'Choose model',
                ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression', 'Naive Bayes', 'Neural Network', 'Compare models']
            )
        test_size = st.slider('Test set size', 0.1, 0.4, 0.2, 0.05)
        st.markdown('---')
        st.write('For housing price, place a CSV file in the project folder named')
        st.write('`housing_price.csv`, `housing-price.csv`, or `data/housing_price.csv`.')
        st.write('If no CSV is found, the app will fall back to a sample housing dataset.')

    if dataset_name == 'Titanic':
        df = load_titanic_data()
        X, y, df_clean = preprocess_titanic(df)

        st.subheader('Titanic dataset overview')
        st.write('Dataset loaded from seaborn Titanic data source.')
        st.dataframe(df_clean.head(10))
        st.write('Rows after cleaning:', len(df_clean))
        st.write('Target class distribution:')
        st.bar_chart(df_clean['survived'].value_counts())

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if model_choice == 'Compare models':
            st.subheader('Model comparison')
            results = {}
            model_list = ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression', 'Naive Bayes', 'Neural Network']
            for selected_model in model_list:
                model = get_classification_model(selected_model)
                results[selected_model] = evaluate_classification_model(model, X_train, X_test, y_train, y_test)

            for model_name, metrics in results.items():
                with st.expander(f'{model_name} metrics'):
                    train_metrics = metrics['train']
                    test_metrics = metrics['test']
                    col1, col2 = st.columns(2)
                    col1.markdown('#### Training metrics')
                    col1.metric('Accuracy', f"{train_metrics['Accuracy']:.4f}")
                    col1.metric('Precision', f"{train_metrics['Precision']:.4f}")
                    col1.metric('Recall', f"{train_metrics['Recall']:.4f}")
                    col1.metric('F1 score', f"{train_metrics['F1 score']:.4f}")
                    col1.write('Training confusion matrix:')
                    col1.dataframe(format_confusion_matrix(train_metrics['Confusion matrix']))

                    col2.markdown('#### Test metrics')
                    col2.metric('Accuracy', f"{test_metrics['Accuracy']:.4f}")
                    col2.metric('Precision', f"{test_metrics['Precision']:.4f}")
                    col2.metric('Recall', f"{test_metrics['Recall']:.4f}")
                    col2.metric('F1 score', f"{test_metrics['F1 score']:.4f}")
                    if 'ROC AUC' in test_metrics and test_metrics['ROC AUC'] is not None:
                        col2.metric('ROC AUC', f"{test_metrics['ROC AUC']:.4f}")
                    if 'Brier score' in test_metrics:
                        col2.metric('Brier score', f"{test_metrics['Brier score']:.4f}")
                    col2.write('Test confusion matrix:')
                    col2.dataframe(format_confusion_matrix(test_metrics['Confusion matrix']))

                    y_prob = test_metrics.get('y_prob')
                    if y_prob is not None:
                        frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
                        fig, ax = plt.subplots()
                        ax.plot(mean_pred, frac_pos, marker='o', label='Calibration')
                        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                        ax.set_xlabel('Mean predicted probability')
                        ax.set_ylabel('Fraction of positives')
                        ax.set_title(f'Calibration curve ({model_name})')
                        ax.legend()
                        st.pyplot(fig)

                        test_df = X_test.copy()
                        test_df['y_true'] = y_test
                        test_df['y_prob'] = y_prob
                        test_df['y_pred'] = (y_prob >= 0.5).astype(int)
                        fp = test_df[(test_df['y_pred'] == 1) & (test_df['y_true'] == 0)].sort_values('y_prob', ascending=False)
                        fn = test_df[(test_df['y_pred'] == 0) & (test_df['y_true'] == 1)].sort_values('y_prob')
                        st.markdown('**Top confident false positives (model too optimistic)**')
                        if not fp.empty:
                            st.dataframe(df_clean.loc[fp.index][['pclass', 'sex', 'age', 'fare']].assign(pred_prob=fp['y_prob'].values).head(5))
                        else:
                            st.write('None')
                        st.markdown('**Top confident false negatives (model too pessimistic)**')
                        if not fn.empty:
                            st.dataframe(df_clean.loc[fn.index][['pclass', 'sex', 'age', 'fare']].assign(pred_prob=fn['y_prob'].values).head(5))
                        else:
                            st.write('None')

            st.markdown('---')
            st.subheader('Test metric comparison chart')
            test_metrics_df = pd.DataFrame(
                {
                    model_name: {
                        'Accuracy': results[model_name]['test']['Accuracy'],
                        'Precision': results[model_name]['test']['Precision'],
                        'Recall': results[model_name]['test']['Recall'],
                        'F1 score': results[model_name]['test']['F1 score'],
                    }
                    for model_name in results
                }
            )
            st.line_chart(test_metrics_df)
        else:
            st.subheader(f'Run results for {model_choice}')
            model = get_classification_model(model_choice)
            metrics = evaluate_classification_model(model, X_train, X_test, y_train, y_test)

            st.markdown('### Training metrics')
            st.metric('Accuracy', f"{metrics['train']['Accuracy']:.4f}")
            st.metric('Precision', f"{metrics['train']['Precision']:.4f}")
            st.metric('Recall', f"{metrics['train']['Recall']:.4f}")
            st.metric('F1 score', f"{metrics['train']['F1 score']:.4f}")
            st.write('Training confusion matrix:')
            st.dataframe(format_confusion_matrix(metrics['train']['Confusion matrix']))

            st.markdown('### Test metrics')
            st.metric('Accuracy', f"{metrics['test']['Accuracy']:.4f}")
            st.metric('Precision', f"{metrics['test']['Precision']:.4f}")
            st.metric('Recall', f"{metrics['test']['Recall']:.4f}")
            st.metric('F1 score', f"{metrics['test']['F1 score']:.4f}")
            st.write('Test confusion matrix:')
            st.dataframe(format_confusion_matrix(metrics['test']['Confusion matrix']))

            st.markdown('### Metrics comparison chart')
            compare_df = pd.DataFrame(
                {
                    'Training': {
                        'Accuracy': metrics['train']['Accuracy'],
                        'Precision': metrics['train']['Precision'],
                        'Recall': metrics['train']['Recall'],
                        'F1 score': metrics['train']['F1 score'],
                    },
                    'Test': {
                        'Accuracy': metrics['test']['Accuracy'],
                        'Precision': metrics['test']['Precision'],
                        'Recall': metrics['test']['Recall'],
                        'F1 score': metrics['test']['F1 score'],
                    }
                }
            )
            st.line_chart(compare_df)

            st.markdown('### Classification report (test set)')
            st.text(metrics['test']['Classification report'])

            y_prob = metrics['test'].get('y_prob')
            if y_prob is not None:
                st.markdown('---')
                st.subheader('Probability-based analysis')
                if metrics['test'].get('ROC AUC') is not None:
                    st.write(f"ROC AUC: {metrics['test']['ROC AUC']:.4f}")
                st.write(f"Brier score: {metrics['test'].get('Brier score'):.4f}")
                frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
                fig, ax = plt.subplots()
                ax.plot(mean_pred, frac_pos, marker='o')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_xlabel('Mean predicted probability')
                ax.set_ylabel('Fraction of positives')
                ax.set_title(f'Calibration curve ({model_choice})')
                st.pyplot(fig)

                test_df = X_test.copy()
                test_df['y_true'] = y_test
                test_df['y_prob'] = y_prob
                test_df['y_pred'] = (y_prob >= 0.5).astype(int)
                fp = test_df[(test_df['y_pred'] == 1) & (test_df['y_true'] == 0)].sort_values('y_prob', ascending=False)
                fn = test_df[(test_df['y_pred'] == 0) & (test_df['y_true'] == 1)].sort_values('y_prob')
                st.markdown('**Top confident false positives (model too optimistic)**')
                if not fp.empty:
                    st.dataframe(df_clean.loc[fp.index][['pclass', 'sex', 'age', 'fare']].assign(pred_prob=fp['y_prob'].values).head(5))
                else:
                    st.write('None')
                st.markdown('**Top confident false negatives (model too pessimistic)**')
                if not fn.empty:
                    st.dataframe(df_clean.loc[fn.index][['pclass', 'sex', 'age', 'fare']].assign(pred_prob=fn['y_prob'].values).head(5))
                else:
                    st.write('None')
    else:
        df = load_housing_data()
        X, y, df_clean, target_col = preprocess_housing(df)

        st.subheader('Housing dataset overview')
        st.write('Loaded from local file if available, otherwise fallback sample housing data.')
        st.write(f'Prediction target: **{target_col}**')
        st.dataframe(df_clean.head(10))
        st.write('Rows after cleaning:', len(df_clean))
        st.write('Target summary statistics:')
        st.write(df_clean[target_col].describe())

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df_clean[target_col].dropna(), bins=30, color='#1f77b4')
        ax.set_title(f'{target_col} distribution')
        ax.set_xlabel(target_col)
        ax.set_ylabel('Count')
        st.pyplot(fig)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        preprocessor = get_housing_preprocessor(X)

        st.subheader(f'Run results for {model_choice}')
        classification_models = ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression', 'Naive Bayes', 'Neural Network']

        if model_choice == 'Compare models':
            st.subheader('Model comparison (classification on price buckets)')
            results = {}
            for selected_model in classification_models:
                model = get_classification_model(selected_model)
                results[selected_model] = evaluate_housing_classifier(preprocessor, X_train, X_test, y_train, y_test, model)

            for model_name, result in results.items():
                with st.expander(f'{model_name} metrics'):
                    test_metrics = result['test']
                    col1, col2 = st.columns(2)
                    col1.markdown('#### Test classification metrics')
                    col1.metric('Accuracy', f"{test_metrics['Accuracy']:.4f}")
                    col1.metric('Precision', f"{test_metrics['Precision']:.4f}")
                    col1.metric('Recall', f"{test_metrics['Recall']:.4f}")
                    col1.metric('F1 score', f"{test_metrics['F1 score']:.4f}")
                    col2.markdown('#### Approx continuous metrics')
                    col2.metric('Approx RMSE', f"{test_metrics['Approx RMSE']:.2f}")
                    col2.metric('Approx MAE', f"{test_metrics['Approx MAE']:.2f}")
                    col2.write('Bin medians:')
                    col2.dataframe(test_metrics['Bin medians'].reset_index(name='Median price').rename(columns={'index': 'Bin'}))

        elif model_choice in classification_models:
            model = get_classification_model(model_choice)
            result = evaluate_housing_classifier(preprocessor, X_train, X_test, y_train, y_test, model)
            st.markdown(f'### {model_choice} price-bucket classification')
            st.write('This option bins the continuous price into buckets and reports classification metrics plus approximate continuous errors.')
            st.metric('Accuracy', f"{result['test']['Accuracy']:.4f}")
            st.metric('Precision', f"{result['test']['Precision']:.4f}")
            st.metric('Recall', f"{result['test']['Recall']:.4f}")
            st.metric('F1 score', f"{result['test']['F1 score']:.4f}")
            st.metric('Approx RMSE', f"{result['test']['Approx RMSE']:.2f}")
            st.metric('Approx MAE', f"{result['test']['Approx MAE']:.2f}")
            st.write('Price bin medians used to approximate continuous prices:')
            st.dataframe(result['bin_medians'].reset_index(name='Median price').rename(columns={'index': 'Bin'}))

            y_pred = result['y_pred']
            analysis_df = X_test.copy()
            analysis_df[target_col] = y_test.values
            analysis_df['Predicted'] = y_pred
            analysis_df['Residual'] = result['residuals']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(analysis_df['Predicted'], analysis_df[target_col], alpha=0.6)
            axes[0].plot([analysis_df['Predicted'].min(), analysis_df['Predicted'].max()], [analysis_df['Predicted'].min(), analysis_df['Predicted'].max()], linestyle='--', color='gray')
            axes[0].set_title('Actual vs Approximate Predicted')
            axes[0].set_xlabel('Approximate Predicted')
            axes[0].set_ylabel('Actual')
            axes[1].hist(analysis_df['Residual'], bins=30, color='#d62728', alpha=0.7)
            axes[1].set_title('Residual distribution')
            axes[1].set_xlabel('Residual')
            plt.tight_layout()
            st.pyplot(fig)

            optimistic = analysis_df[analysis_df['Residual'] < 0].copy()
            pessimistic = analysis_df[analysis_df['Residual'] > 0].copy()
            optimistic = optimistic.assign(ErrorMagnitude=(-optimistic['Residual']))
            pessimistic = pessimistic.assign(ErrorMagnitude=pessimistic['Residual'].abs())

            st.markdown('**Top optimistic errors (predicted too high)**')
            if not optimistic.empty:
                st.dataframe(optimistic.sort_values('ErrorMagnitude', ascending=False).head(5)[['Predicted', target_col, 'Residual', 'ErrorMagnitude']])
            else:
                st.write('None')

            st.markdown('**Top pessimistic errors (predicted too low)**')
            if not pessimistic.empty:
                st.dataframe(pessimistic.sort_values('ErrorMagnitude', ascending=False).head(5)[['Predicted', target_col, 'Residual', 'ErrorMagnitude']])
            else:
                st.write('None')

        else:
            # Regression models (continuous prediction)
            model = get_regression_model(model_choice)
            result = evaluate_regression_model(model, preprocessor, X_train, X_test, y_train, y_test)

            st.markdown('### Training metrics')
            st.metric('RMSE', f"{result['train']['RMSE']:.2f}")
            st.metric('MAE', f"{result['train']['MAE']:.2f}")
            st.metric('R2', f"{result['train']['R2']:.4f}")

            st.markdown('### Test metrics')
            st.metric('RMSE', f"{result['test']['RMSE']:.2f}")
            st.metric('MAE', f"{result['test']['MAE']:.2f}")
            st.metric('R2', f"{result['test']['R2']:.4f}")

            compare_df = pd.DataFrame({
                'Training': result['train'],
                'Test': result['test'],
            })
            st.line_chart(compare_df)

            y_pred = result['y_pred']
            analysis_df = X_test.copy()
            analysis_df[target_col] = y_test.values
            analysis_df['Predicted'] = y_pred
            analysis_df['Residual'] = result['residuals']

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].scatter(analysis_df['Predicted'], analysis_df[target_col], alpha=0.6)
            axes[0].plot([analysis_df['Predicted'].min(), analysis_df['Predicted'].max()], [analysis_df['Predicted'].min(), analysis_df['Predicted'].max()], linestyle='--', color='gray')
            axes[0].set_title('Actual vs Predicted')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            axes[1].hist(analysis_df['Residual'], bins=30, color='#d62728', alpha=0.7)
            axes[1].set_title('Residual distribution')
            axes[1].set_xlabel('Residual')
            plt.tight_layout()
            st.pyplot(fig)

            optimistic = analysis_df[analysis_df['Residual'] < 0].copy()
            pessimistic = analysis_df[analysis_df['Residual'] > 0].copy()
            optimistic = optimistic.assign(ErrorMagnitude=(-optimistic['Residual']))
            pessimistic = pessimistic.assign(ErrorMagnitude=pessimistic['Residual'].abs())

            st.markdown('**Top optimistic errors (predicted too high)**')
            if not optimistic.empty:
                st.dataframe(optimistic.sort_values('ErrorMagnitude', ascending=False).head(5)[['Predicted', target_col, 'Residual', 'ErrorMagnitude']])
            else:
                st.write('None')

            st.markdown('**Top pessimistic errors (predicted too low)**')
            if not pessimistic.empty:
                st.dataframe(pessimistic.sort_values('ErrorMagnitude', ascending=False).head(5)[['Predicted', target_col, 'Residual', 'ErrorMagnitude']])
            else:
                st.write('None')

    st.sidebar.markdown('---')
    st.sidebar.write('Streamlit UI for dataset selection and model comparison.')


if __name__ == '__main__':
    main()

# HW: fix all the errors in this. Provide summary of insights for housing dataset, which model worked best, how did I preprocess data points. 
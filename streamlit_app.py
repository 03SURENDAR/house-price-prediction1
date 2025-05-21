import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

st.set_page_config(page_title='House Price Prediction Dashboard', layout='wide')

st.title('🏠 House Price Prediction Dashboard')
st.write("""This interactive app lets you **explore, train and deploy** smart regression models
to forecast house prices accurately. Start by loading the dataset, run Exploratory Data Analysis (EDA),
train a model, inspect metrics & feature importances, then predict prices for new houses.""")

# --- Sidebar Controls --------------------------------------------------------
st.sidebar.header('1. Load Dataset')
data_source = st.sidebar.radio('Choose data source', ('Example dataset from repo', 'Upload CSV'))
if data_source == 'Example dataset from repo':
    DATA_URL = 'data/housing.csv'
    if os.path.exists(DATA_URL):
        df = pd.read_csv(DATA_URL)
    else:
        st.sidebar.error('Example dataset not found. Please upload a CSV instead.')
        df = None
else:
    uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = None

if df is not None:
    st.subheader('Dataset Preview')
    st.dataframe(df.head())

    # Identify numeric & categorical columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    if 'SalePrice' in num_cols:
        default_target_index = num_cols.index('SalePrice')
    else:
        default_target_index = 0
    target_col = st.sidebar.selectbox('Select target column (house price)', options=num_cols, index=default_target_index)

    # --- EDA -----------------------------------------------------------------
    st.header('2. Exploratory Data Analysis')
    with st.expander('Show Summary Statistics'):
        st.write(df.describe())

    with st.expander('Show Correlation Heatmap'):
        import plotly.express as px
        corr = df[num_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', title='Correlation Matrix (Numeric Features)')
        st.plotly_chart(fig, use_container_width=True)

    # --- Model Training ------------------------------------------------------
    st.header('3. Model Training')
    model_choice = st.selectbox('Select algorithm', [
        'LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'GradientBoosting'] + (['XGBoost'] if xgb_available else []))

    test_size = st.slider('Test size (fraction)', 0.1, 0.4, 0.2, 0.05)

    if model_choice == 'RandomForest':
        n_estimators = st.number_input('n_estimators', 100, 1000, 300, 50)
    elif model_choice in ['GradientBoosting', 'XGBoost']:
        n_estimators = st.number_input('n_estimators', 100, 1000, 200, 50)
        learning_rate = st.number_input('learning_rate', 0.01, 0.5, 0.1, 0.01)

    train_button = st.button('Train Model')
    if train_button:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ], remainder='drop')

        if model_choice == 'LinearRegression':
            model = LinearRegression()
        elif model_choice == 'Ridge':
            model = Ridge(alpha=1.0)
        elif model_choice == 'Lasso':
            model = Lasso(alpha=0.001)
        elif model_choice == 'RandomForest':
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_choice == 'GradientBoosting':
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        elif model_choice == 'XGBoost':
            model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, objective='reg:squarederror', random_state=42)

        pipeline = Pipeline(steps=[('pre', preprocessor), ('model', model)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        st.subheader('Performance Metrics')
        st.write({'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2})

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_names = preprocessor.get_feature_names_out()
            imp_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values(by='importance', ascending=False).head(20)
            st.subheader('Top Feature Importances')
            st.bar_chart(imp_df.set_index('feature'))

        # Save the trained pipeline
        joblib.dump(pipeline, 'house_price_model.pkl')
        st.success('Model trained and saved as house_price_model.pkl')

    # --- Prediction ----------------------------------------------------------
    st.header('4. Predict House Price')
    if os.path.exists('house_price_model.pkl'):
        pipeline = joblib.load('house_price_model.pkl')
        st.info('Using saved model for prediction.')
        with st.form('prediction_form'):
            st.write('Input features to predict {}:'.format(target_col))
            input_data = {}
            for col in num_cols:
                if col == target_col:
                    continue
                val = st.number_input(f'{col}', value=float(df[col].median()))
                input_data[col] = val
            for col in cat_cols:
                val = st.text_input(f'{col}', value=str(df[col].mode()[0]))
                input_data[col] = val
            submitted = st.form_submit_button('Predict')
            if submitted:
                input_df = pd.DataFrame([input_data])
                prediction = pipeline.predict(input_df)[0]
                st.subheader(f'Predicted {target_col}: {prediction:,.2f}')
    else:
        st.info('Train and save a model to enable predictions.')

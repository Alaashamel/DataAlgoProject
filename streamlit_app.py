import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Telco Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        margin-bottom: 1.5rem;
    }
    
    /* Cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3498db;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Load custom fonts
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    * {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# Function to calculate missing values
def missing_summary(df):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percent
    })
    return missing_df

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # Clean data: convert numeric columns
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = load_data()
    st.session_state.df_processed = None
    st.session_state.show_raw_data = False

# Main title with gradient effect
st.markdown('<h1 class="main-header">📊 Telco Customer Churn Analytics Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Navigation
    st.markdown("### Navigation")
    page = st.radio(
        "Select Page",
        ["📈 Overview", "🔍 Data Explorer", "🧹 Data Processing", "📊 Visual Analytics", "🎯 Churn Insights", "🚀 Model Ready Data"],
        index=0
    )
    
    # Filters
    st.markdown("### 🔍 Data Filters")
    
    # Load filters
    df_raw = st.session_state.df
    gender_options = df_raw['gender'].unique()
    contract_options = df_raw['Contract'].unique()
    churn_options = df_raw['Churn'].unique()
    
    gender_filter = st.multiselect("Gender", options=gender_options, default=gender_options)
    contract_filter = st.multiselect("Contract Type", options=contract_options, default=contract_options)
    churn_filter = st.multiselect("Churn Status", options=churn_options, default=churn_options)
    
    # Apply filters
    df_filtered = df_raw[
        df_raw['gender'].isin(gender_filter) &
        df_raw['Contract'].isin(contract_filter) &
        df_raw['Churn'].isin(churn_filter)
    ]
    
    # Statistics
    st.markdown("### 📊 Dataset Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df_filtered))
    with col2:
        st.metric("Features", df_filtered.shape[1])
    
    # Theme selector
    st.markdown("### 🎨 Theme")
    theme = st.selectbox("Chart Theme", ["plotly", "plotly_white", "plotly_dark", "seaborn", "ggplot2"])
    
    # Download processed data
    if st.session_state.df_processed is not None:
        csv = st.session_state.df_processed.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed Data",
            data=csv,
            file_name="telco_churn_processed.csv",
            mime="text/csv"
        )

# Page 1: Overview
if page == "📈 Overview":
    st.markdown('<h2 class="section-header">📈 Executive Overview</h2>', unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown("### 📊 Key Business Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df_filtered)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_customers:,}</div>
            <div class="metric-label">Total Customers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_rate = (df_filtered['Churn'] == 'Yes').mean() * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-label">Churn Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_tenure = df_filtered['tenure'].mean()
        if pd.isna(avg_tenure):
            avg_tenure = 0  # or some default
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_tenure:.1f}</div>
            <div class="metric-label">Avg Tenure (Months)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_monthly_charge = df_filtered['MonthlyCharges'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_monthly_charge:.2f}</div>
            <div class="metric-label">Avg Monthly Charge</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Project Description
    st.markdown("### 🎯 Project Objective")
    st.info("""
    **Business Goal:** Reduce customer churn by 25% in the next 6 months through data-driven insights.
    
    **Analytical Approach:**
    1. Exploratory Data Analysis to understand churn patterns
    2. Data preprocessing and feature engineering
    3. Predictive modeling for churn risk assessment
    4. Actionable insights for retention strategies
    """)
    
    # Quick Insights
    st.markdown("### 💡 Quick Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by Contract Type
        contract_churn = df_filtered.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        fig = go.Figure(data=[
            go.Bar(x=contract_churn.index, y=contract_churn.values,
                   marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                   text=contract_churn.values.round(1),
                   texttemplate='%{text}%',
                   textposition='outside')
        ])
        fig.update_layout(
            title="Churn Rate by Contract Type",
            template=theme,
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn by Internet Service
        internet_churn = df_filtered.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        fig = go.Figure(data=[
            go.Pie(labels=internet_churn.index, values=internet_churn.values,
                   hole=0.4,
                   marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ])
        fig.update_layout(
            title="Churn Distribution by Internet Service",
            template=theme,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# Page 2: Data Explorer
elif page == "🔍 Data Explorer":
    st.markdown('<h2 class="section-header">🔍 Data Exploration & Profiling</h2>', unsafe_allow_html=True)
    
    # Data Preview
    tabs = st.tabs(["📋 Data Preview", "📊 Statistics", "🔍 Missing Values", "📈 Distributions"])
    
    with tabs[0]:
        st.markdown("### Dataset Preview")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            num_rows = st.slider("Number of rows to display", 5, 50, 10)
            st.dataframe(df_filtered.head(num_rows), use_container_width=True)
        
        with col2:
            st.markdown("#### Shape Info")
            st.metric("Rows", df_filtered.shape[0])
            st.metric("Columns", df_filtered.shape[1])
            
            st.markdown("#### Data Types")
            dtypes = df_filtered.dtypes.value_counts()
            for dtype, count in dtypes.items():
                st.text(f"{dtype}: {count}")
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Numerical Statistics")
            st.dataframe(df_filtered.describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)
        
        with col2:
            st.markdown("#### Categorical Statistics")
            cat_stats = df_filtered.describe(include='O').T
            st.dataframe(cat_stats.style.background_gradient(cmap='Greens'), use_container_width=True)
    
    with tabs[2]:
        st.markdown("#### Missing Value Analysis")
        missing_df = missing_summary(df_filtered)
        
        # Only show columns with missing values
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(missing_df.style.background_gradient(cmap='Reds'), use_container_width=True)
            
            with col2:
                fig = px.bar(missing_df, x=missing_df.index, y='Missing Percentage',
                            title="Missing Values by Column",
                            color='Missing Percentage',
                            color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No missing values found in the dataset!")
    
    with tabs[3]:
        st.markdown("#### Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", df_filtered.select_dtypes(include=[np.number]).columns)
            
            fig = px.histogram(df_filtered, x=feature, nbins=50,
                              title=f"Distribution of {feature}",
                              color_discrete_sequence=['#3498db'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            cat_feature = st.selectbox("Select Categorical Feature", df_filtered.select_dtypes(include=['object']).columns)
            
            value_counts = df_filtered[cat_feature].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Top 10 Values for {cat_feature}",
                        color=value_counts.values,
                        color_continuous_scale='Viridis')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Page 3: Data Processing
elif page == "🧹 Data Processing":
    st.markdown('<h2 class="section-header">🧹 Data Preprocessing Pipeline</h2>', unsafe_allow_html=True)
    
    # Initialize processed dataframe
    if st.session_state.df_processed is None:
        st.session_state.df_processed = df_filtered.copy()
    
    # Processing Steps
    processing_steps = st.multiselect(
        "Select Processing Steps to Apply",
        ["Convert TotalCharges", "Handle Missing Values", "Standardize Text", 
         "Remove Duplicates", "Handle Outliers", "Encode Categorical Variables"],
        default=["Convert TotalCharges", "Standardize Text", "Encode Categorical Variables"]
    )
    
    # Process data
    df_processed = df_filtered.copy()
    
    with st.expander("Processing Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if "Convert TotalCharges" in processing_steps:
                st.markdown("#### 🔄 Convert TotalCharges and Tenure")
                df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
                df_processed['tenure'] = pd.to_numeric(df_processed['tenure'], errors='coerce')
                st.success("✅ Converted TotalCharges and tenure to numeric")

                # Show conversion stats
                missing_total = df_processed['TotalCharges'].isnull().sum()
                missing_tenure = df_processed['tenure'].isnull().sum()
                if missing_total > 0:
                    st.warning(f"⚠️ {missing_total} TotalCharges values converted to NaN")
                if missing_tenure > 0:
                    st.warning(f"⚠️ {missing_tenure} tenure values converted to NaN")
            
            if "Standardize Text" in processing_steps:
                st.markdown("#### ✨ Standardize Text")
                text_cols = ['PaymentMethod', 'Contract', 'InternetService']
                for col in text_cols:
                    df_processed[col] = df_processed[col].str.strip().str.lower().str.replace('-', ' ').str.title()
                st.success(f"✅ Standardized {len(text_cols)} text columns")
                
                # Show before/after
                if st.checkbox("Show text standardization example"):
                    sample = df_filtered[text_cols].iloc[0].to_dict()
                    st.json({"Before": sample})
        
        with col2:
            if "Remove Duplicates" in processing_steps:
                st.markdown("#### 🗑️ Remove Duplicates")
                initial_rows = len(df_processed)
                df_processed.drop_duplicates(inplace=True)
                removed = initial_rows - len(df_processed)
                if removed > 0:
                    st.warning(f"⚠️ Removed {removed} duplicate rows")
                else:
                    st.success("✅ No duplicates found")
            
            if "Handle Outliers" in processing_steps:
                st.markdown("#### 📊 Handle Outliers")
                numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
                
                for col in numeric_cols:
                    if col in df_processed.columns:
                        Q1 = df_processed[col].quantile(0.25)
                        Q3 = df_processed[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        outliers = df_processed[(df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)]
                        st.text(f"{col}: {len(outliers)} outliers detected")
    
    # Encoding Section
    if "Encode Categorical Variables" in processing_steps:
        st.markdown("#### 🔤 Categorical Encoding")
        
        encoding_tabs = st.tabs(["Binary Encoding", "One-Hot Encoding", "Result Preview"])
        
        with encoding_tabs[0]:
            st.markdown("##### Binary Variables (0/1)")
            binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
            
            binary_mapping = {
                'gender': {'Female': 0, 'Male': 1},
                'Partner': {'No': 0, 'Yes': 1},
                'Dependents': {'No': 0, 'Yes': 1},
                'PhoneService': {'No': 0, 'Yes': 1},
                'PaperlessBilling': {'No': 0, 'Yes': 1},
                'Churn': {'No': 0, 'Yes': 1}
            }
            
            for col in binary_cols:
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].map(binary_mapping.get(col, {}))
            
            st.success(f"✅ Encoded {len(binary_cols)} binary variables")
            
            # Show sample
            st.dataframe(df_processed[binary_cols].head())
        
        with encoding_tabs[1]:
            st.markdown("##### One-Hot Encoding")
            
            # Multi-binary columns
            multi_binary_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            
            # Keep only existing columns
            existing_cols = [col for col in multi_binary_cols if col in df_processed.columns]
            
            if existing_cols:
                df_processed = pd.get_dummies(df_processed, columns=existing_cols, drop_first=True, dtype=int)
                st.success(f"✅ One-hot encoded {len(existing_cols)} multi-binary columns")
            
            # Nominal columns
            nominal_cols = ['InternetService', 'Contract', 'PaymentMethod']
            df_processed = pd.get_dummies(df_processed, columns=nominal_cols, drop_first=True, dtype=int)
            st.success(f"✅ One-hot encoded {len(nominal_cols)} nominal columns")
        
        with encoding_tabs[2]:
            st.markdown("##### Final Processed Data")
            st.dataframe(df_processed.head(10))
            st.metric("Final Shape", f"{df_processed.shape[0]} rows × {df_processed.shape[1]} columns")
    
    # Save processed data
    if st.button("💾 Save Processed Data"):
        st.session_state.df_processed = df_processed.copy()
        st.success("✅ Processed data saved to session!")
        
        # Show differences
        original_cols = set(df_filtered.columns)
        processed_cols = set(df_processed.columns)
        new_cols = processed_cols - original_cols
        
        st.info(f"Created {len(new_cols)} new features through encoding")

# Page 4: Visual Analytics
elif page == "📊 Visual Analytics":
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; color: white;'>
    <h2>📊 Interactive Visualizations</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Use processed or raw data
    df_viz = st.session_state.df_processed if st.session_state.df_processed is not None else df_filtered

    # Helper function for churn mean calculation
    def churn_mean(x):
        if x.dtype == 'object':
            return (x == 'Yes').mean()
        else:
            return x.mean()

    # Visualization tabs
    viz_tabs = st.tabs(["📈 Correlation Analysis", "🎯 Churn Analysis", "📊 Feature Distributions", "🔗 Relationships"])
    
    with viz_tabs[0]:
        st.markdown("#### Correlation Matrix")
        
        # Select numeric columns
        numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect("Select columns for correlation", numeric_cols, default=numeric_cols[:8])
        
        if len(selected_cols) >= 2:
            corr_matrix = df_viz[selected_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title="Correlation Heatmap",
                height=600,
                template=theme
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlations
            st.markdown("#### Top Correlations")
            corr_series = corr_matrix.unstack()
            corr_series = corr_series[corr_series < 1].sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Correlations:**")
                top_pos = corr_series.head(5)
                for idx, value in top_pos.items():
                    st.text(f"{idx[0]} ↔ {idx[1]}: {value:.3f}")
            
            with col2:
                st.markdown("**Top Negative Correlations:**")
                top_neg = corr_series.tail(5)[::-1]
                for idx, value in top_neg.items():
                    st.text(f"{idx[0]} ↔ {idx[1]}: {value:.3f}")
    
    with viz_tabs[1]:
        st.markdown("#### Churn Analysis Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            if 'Churn' in df_viz.columns:
                churn_counts = df_viz['Churn'].value_counts()
                if isinstance(churn_counts.index[0], (int, float)):
                    labels = ['Retained' if x == 0 else 'Churned' for x in churn_counts.index]
                else:
                    labels = churn_counts.index
                
                fig = px.pie(values=churn_counts.values, names=labels,
                            title="Customer Churn Distribution",
                            color_discrete_sequence=['#2ecc71', '#e74c3c'],
                            hole=0.3)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Churn over time (using tenure as proxy)
            if 'tenure' in df_viz.columns and 'Churn' in df_viz.columns:
                tenure_churn = df_viz.groupby('tenure')['Churn'].apply(churn_mean).reset_index()

                fig = px.line(tenure_churn, x='tenure', y='Churn',
                              title="Churn Rate by Tenure",
                              markers=True)
                fig.update_layout(height=400)
                fig.update_traces(line_color='#e74c3c', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
        
        # Churn by features
        st.markdown("#### Churn by Different Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            feature = st.selectbox("Select Feature 1", 
                                  [col for col in df_viz.columns if df_viz[col].nunique() < 10],
                                  key='feature1')
            
            if feature in df_viz.columns:
                churn_by_feature = df_viz.groupby(feature)['Churn'].apply(churn_mean).reset_index()
                fig = px.bar(churn_by_feature, x=feature, y='Churn',
                             title=f"Churn Rate by {feature}",
                             color='Churn',
                             color_continuous_scale='Reds')
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            feature2 = st.selectbox("Select Feature 2", 
                                   [col for col in df_viz.columns if df_viz[col].nunique() < 10],
                                   key='feature2')
            
            if feature2 in df_viz.columns:
                # Use countplot style
                churn_counts = df_viz.groupby([feature2, 'Churn']).size().reset_index(name='count')
                fig = px.bar(churn_counts, x=feature2, y='count', color='Churn',
                            title=f"Distribution by {feature2} and Churn",
                            barmode='group',
                            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Box plot for numeric features
            numeric_features = df_viz.select_dtypes(include=[np.number]).columns.tolist()
            if 'Churn' in numeric_features:
                numeric_features.remove('Churn')
            
            selected_num = st.selectbox("Select Numeric Feature", numeric_features, key='num_feature')
            
            if selected_num in df_viz.columns:
                fig = px.box(df_viz, x='Churn', y=selected_num,
                            title=f"{selected_num} Distribution by Churn",
                            color='Churn',
                            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:
        st.markdown("#### Feature Distributions")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            plot_type = st.selectbox("Plot Type", ["Histogram", "Box Plot", "Violin Plot", "Density Plot"])
            feature = st.selectbox("Select Feature", df_viz.columns)
            
            if plot_type == "Histogram":
                bins = st.slider("Number of bins", 10, 100, 30)
        
        with col2:
            if feature in df_viz.columns:
                if plot_type == "Histogram":
                    fig = px.histogram(df_viz, x=feature, nbins=bins,
                                      title=f"Distribution of {feature}",
                                      color_discrete_sequence=['#2c3e50'])

                elif plot_type == "Box Plot":
                    fig = px.box(df_viz, y=feature,
                                title=f"Box Plot of {feature}",
                                color_discrete_sequence=['#34495e'])

                elif plot_type == "Violin Plot":
                    fig = px.violin(df_viz, y=feature,
                                   title=f"Violin Plot of {feature}",
                                   color_discrete_sequence=['#16a085'])

                elif plot_type == "Density Plot":
                    fig = px.histogram(df_viz, x=feature, nbins=50,
                                      histnorm='density',
                                      title=f"Density Plot of {feature}",
                                      color_discrete_sequence=['#229954'])
                
                fig.update_layout(height=500, template=theme)
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        st.markdown("#### Feature Relationships")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis Feature", 
                                    df_viz.select_dtypes(include=[np.number]).columns,
                                    key='x_feature')
        
        with col2:
            y_feature = st.selectbox("Y-axis Feature", 
                                    df_viz.select_dtypes(include=[np.number]).columns,
                                    key='y_feature')
        
        if x_feature != y_feature:
            fig = px.scatter(df_viz, x=x_feature, y=y_feature,
                            title=f"{x_feature} vs {y_feature}",
                            opacity=0.6,
                            color_discrete_sequence=['#3498db'])
            
            fig.update_layout(height=500, template=theme)
            st.plotly_chart(fig, use_container_width=True)

# Page 5: Churn Insights
elif page == "🎯 Churn Insights":
    st.markdown('<h2 class="section-header">🎯 Actionable Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    # Business Insights
    st.markdown("### 📋 Executive Summary")
    
    # Calculate key metrics
    total_customers = len(df_filtered)
    churn_rate = (df_filtered['Churn'] == 'Yes').mean() * 100
    avg_monthly_rev = df_filtered['MonthlyCharges'].mean()
    total_monthly_rev_loss = avg_monthly_rev * (total_customers * churn_rate / 100)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        #### 🎯 Key Findings
        
        1. **High-Risk Customers**
           - Month-to-month contract customers have highest churn
           - Fiber optic internet users churn more frequently
           - Electronic check payment method correlates with churn
        
        2. **Protective Factors**
           - Longer tenure customers are less likely to churn
           - Customers with tech support churn less
           - Annual/Two-year contracts have lower churn rates
        
        3. **Revenue Impact**
           - Churning customers cost ~$X monthly in lost revenue
           - High-value customers (high monthly charges) are at risk
        """)
    
    with insights_col2:
        st.markdown("""
        #### 💡 Recommendations
        
        **Immediate Actions (Next 30 days):**
        1. **Targeted Retention Campaign** for month-to-month customers
        2. **Promote Annual Contracts** with incentives
        3. **Improve Tech Support** accessibility and quality
        
        **Medium-term (Next 90 days):**
        1. **Develop Early Warning System** for at-risk customers
        2. **Personalized Offers** based on usage patterns
        3. **Payment Method Optimization** - encourage auto-pay
        
        **Long-term Strategy:**
        1. **Product Improvement** for fiber optic service
        2. **Customer Success Program** for high-value clients
        3. **Predictive Modeling** for proactive retention
        """)
    
    # Detailed Analysis
    st.markdown("""
    <div style='background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem; color: white;'>
    <h3>📊 Detailed Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    analysis_tabs = st.tabs(["📈 Churn Drivers", "💰 Financial Impact", "👥 Customer Segments"])
    
    with analysis_tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top churn drivers
            st.markdown("#### Top 5 Churn Drivers")
            
            # Calculate churn rates by various features
            drivers = []
            
            # Contract type
            if 'Contract' in df_filtered.columns:
                contract_churn = df_filtered.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                drivers.append(("Contract: Month-to-month", contract_churn.max()))
            
            # Internet service
            if 'InternetService' in df_filtered.columns:
                internet_churn = df_filtered.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                drivers.append(("Internet: Fiber Optic", internet_churn.max()))
            
            # Payment method
            if 'PaymentMethod' in df_filtered.columns:
                payment_churn = df_filtered.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                drivers.append(("Payment: Electronic Check", payment_churn.max()))
            
            # Paperless billing
            if 'PaperlessBilling' in df_filtered.columns:
                paperless_churn = df_filtered.groupby('PaperlessBilling')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                drivers.append(("Paperless Billing: Yes", paperless_churn.max()))
            
            # Tech support
            if 'TechSupport' in df_filtered.columns:
                tech_churn = df_filtered.groupby('TechSupport')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
                drivers.append(("Tech Support: No", tech_churn.max()))
            
            # Display as bar chart
            drivers_df = pd.DataFrame(drivers, columns=['Driver', 'Churn Rate'])
            drivers_df = drivers_df.sort_values('Churn Rate', ascending=False).head(5)
            
            fig = px.bar(drivers_df, x='Churn Rate', y='Driver',
                        orientation='h',
                        title="Top Churn Risk Factors",
                        color='Churn Rate',
                        color_continuous_scale='Reds')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Retention opportunities
            st.markdown("#### Retention Opportunities")
            
            opportunities = [
                ("Convert to Annual Contracts", "30% potential churn reduction"),
                ("Add Tech Support", "25% reduction for at-risk customers"),
                ("Promote Auto-pay", "20% reduction in payment-related churn"),
                ("Tenure-based Loyalty Program", "15% improved retention"),
                ("Personalized Offers", "10% immediate churn reduction")
            ]
            
            for i, (opp, impact) in enumerate(opportunities, 1):
                st.markdown(f"""
                <div style='background: linear-gradient(45deg, #2c3e50, #34495e);
                            padding: 1rem;
                            margin-bottom: 0.5rem;
                            border-radius: 5px;
                            border-left: 4px solid #e74c3c;
                            color: white;'>
                    <strong>#{i}: {opp}</strong><br>
                    <small>📈 {impact}</small>
                </div>
                """, unsafe_allow_html=True)
    
    with analysis_tabs[1]:
        st.markdown("#### 💰 Financial Impact Analysis")
        
        # Calculate potential revenue at risk
        high_risk_customers = df_filtered[
            (df_filtered['Contract'] == 'Month-to-month') &
            (df_filtered['InternetService'] == 'Fiber optic') &
            (df_filtered['PaymentMethod'] == 'Electronic check')
        ]
        
        at_risk_revenue = high_risk_customers['MonthlyCharges'].sum()
        total_revenue = df_filtered['MonthlyCharges'].sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High-Risk Revenue", f"${at_risk_revenue:,.0f}/month")
        
        with col2:
            st.metric("Total Monthly Revenue", f"${total_revenue:,.0f}/month")
        
        with col3:
            risk_percentage = (at_risk_revenue / total_revenue) * 100
            st.metric("Revenue at Risk", f"{risk_percentage:.1f}%")
        
        # ROI calculation
        st.markdown("#### 📈 ROI Projection")
        
        retention_cost = st.slider("Estimated Retention Cost per Customer ($)", 10, 100, 25)
        expected_improvement = st.slider("Expected Churn Reduction (%)", 5, 50, 20)
        
        current_churn_count = (df_filtered['Churn'] == 'Yes').sum()
        potential_savings = current_churn_count * (expected_improvement / 100) * avg_monthly_rev
        total_cost = len(high_risk_customers) * retention_cost
        roi = ((potential_savings - total_cost) / total_cost) * 100
        
        st.info(f"""
        **ROI Analysis:**
        - **Monthly Savings Potential:** ${potential_savings:,.0f}
        - **Retention Program Cost:** ${total_cost:,.0f}
        - **Projected ROI:** {roi:.0f}%
        - **Breakeven:** {total_cost / (potential_savings / 12):.1f} months
        """)
    
    with analysis_tabs[2]:
        st.markdown("#### 👥 Customer Segmentation")
        
        # Create segments based on risk
        df_filtered['Risk_Score'] = 0
        
        # Add risk points
        conditions = [
            (df_filtered['Contract'] == 'Month-to-month', 3),
            (df_filtered['InternetService'] == 'Fiber optic', 2),
            (df_filtered['PaymentMethod'] == 'Electronic check', 2),
            (df_filtered['TechSupport'] == 'No', 1),
            (df_filtered['PaperlessBilling'] == 'Yes', 1)
        ]
        
        for condition, points in conditions:
            df_filtered.loc[condition, 'Risk_Score'] += points
        
        # Define segments
        df_filtered['Segment'] = pd.cut(df_filtered['Risk_Score'],
                                       bins=[-1, 1, 3, 10],
                                       labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        segment_stats = df_filtered.groupby('Segment').agg({
            'Churn': lambda x: (x == 'Yes').mean() * 100,
            'MonthlyCharges': 'mean',
            'tenure': 'mean',
            'customerID': 'count'
        }).round(2)
        
        segment_stats.columns = ['Churn Rate %', 'Avg Monthly Charge', 'Avg Tenure', 'Customer Count']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(segment_stats.style.background_gradient(cmap='YlOrRd', subset=['Churn Rate %']))
        
        with col2:
            fig = px.sunburst(df_filtered, path=['Segment', 'Contract', 'InternetService'],
                             title="Customer Segments by Contract & Internet",
                             color='Risk_Score',
                             color_continuous_scale='Reds')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# Page 6: Model Ready Data
elif page == "🚀 Model Ready Data":
    st.markdown('<h2 class="section-header">🚀 Machine Learning Preparation</h2>', unsafe_allow_html=True)
    
    # Use processed data or process now
    if st.session_state.df_processed is None:
        st.warning("⚠️ Processed data not found. Please run data processing first.")
        
        if st.button("🔄 Process Data Now"):
            # Simple processing
            df_processed = df_filtered.copy()
            df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
            
            # Binary encoding
            binary_mapping = {
                'gender': {'Female': 0, 'Male': 1},
                'Partner': {'No': 0, 'Yes': 1},
                'Dependents': {'No': 0, 'Yes': 1},
                'PhoneService': {'No': 0, 'Yes': 1},
                'PaperlessBilling': {'No': 0, 'Yes': 1},
                'Churn': {'No': 0, 'Yes': 1}
            }
            
            for col, mapping in binary_mapping.items():
                if col in df_processed.columns:
                    df_processed[col] = df_processed[col].map(mapping)
            
            # One-hot encoding
            categorical_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                              'DeviceProtection', 'TechSupport', 'StreamingTV',
                              'StreamingMovies', 'InternetService', 'Contract', 'PaymentMethod']
            
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True, dtype=int)
            
            st.session_state.df_processed = df_processed
            st.success("✅ Data processed successfully!")
    
    if st.session_state.df_processed is not None:
        df_ml = st.session_state.df_processed
        
        # ML Preparation Dashboard
        st.markdown("### 📋 Model Features Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Features", df_ml.shape[1])
        
        with col2:
            numeric_features = len(df_ml.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", numeric_features)
        
        with col3:
            categorical_features = len(df_ml.select_dtypes(include=['object', 'category']).columns)
            st.metric("Categorical Features", categorical_features)
        
        # Feature importance preview
        st.markdown("### 🎯 Target Variable Analysis")
        
        if 'Churn' in df_ml.columns:
            target_distribution = df_ml['Churn'].value_counts(normalize=True) * 100
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Class Distribution")
                for value, percentage in target_distribution.items():
                    label = "Churned" if value == 1 else "Retained"
                    st.progress(int(percentage), text=f"{label}: {percentage:.1f}%")
                
                imbalance_ratio = target_distribution.max() / target_distribution.min()
                st.metric("Class Imbalance Ratio", f"{imbalance_ratio:.2f}:1")
            
            with col2:
                # Feature correlation with target
                numeric_cols_ml = df_ml.select_dtypes(include=[np.number]).columns.tolist()
                if 'Churn' in numeric_cols_ml:
                    numeric_cols_ml.remove('Churn')

                # Ensure Churn is numeric
                if df_ml['Churn'].dtype == 'object':
                    df_ml['Churn'] = (df_ml['Churn'] == 'Yes').astype(int)

                if numeric_cols_ml:
                    correlations = df_ml[numeric_cols_ml + ['Churn']].corr()['Churn'].drop('Churn').sort_values()
                    
                    fig = px.bar(x=correlations.values, y=correlations.index,
                                orientation='h',
                                title="Feature Correlation with Churn",
                                color=correlations.values,
                                color_continuous_scale='RdBu',
                                color_continuous_midpoint=0)
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Data splitting
        st.markdown("### 🪓 Train-Test Split Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20)
            random_state = st.number_input("Random State", 0, 100, 42)
        
        with col2:
            shuffle_data = st.checkbox("Shuffle Data", value=True)
            stratify_split = st.checkbox("Stratified Split (maintain class ratio)", value=True)
        
        if st.button("🔧 Prepare Train/Test Split"):
            from sklearn.model_selection import train_test_split
            
            if 'Churn' in df_ml.columns:
                X = df_ml.drop('Churn', axis=1)
                y = df_ml['Churn']
                
                if stratify_split:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, 
                        test_size=test_size/100,
                        random_state=random_state,
                        shuffle=shuffle_data,
                        stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size/100,
                        random_state=random_state,
                        shuffle=shuffle_data
                    )
                
                st.success(f"✅ Data split successfully!")
                
                # Show split statistics
                split_stats = pd.DataFrame({
                    'Dataset': ['Training', 'Testing', 'Total'],
                    'Samples': [len(X_train), len(X_test), len(X)],
                    'Churn Rate %': [
                        y_train.mean() * 100,
                        y_test.mean() * 100,
                        y.mean() * 100
                    ]
                })
                
                st.dataframe(split_stats)
                
                # Visualize split
                fig = make_subplots(rows=1, cols=2,
                                  subplot_titles=("Training Set", "Testing Set"),
                                  specs=[[{'type': 'domain'}, {'type': 'domain'}]])
                
                # Training set
                train_counts = y_train.value_counts()
                fig.add_trace(go.Pie(labels=['Retained', 'Churned'],
                                    values=train_counts.values,
                                    hole=0.3,
                                    marker_colors=['#2ecc71', '#e74c3c'],
                                    name="Training"), 1, 1)
                
                # Testing set
                test_counts = y_test.value_counts()
                fig.add_trace(go.Pie(labels=['Retained', 'Churned'],
                                    values=test_counts.values,
                                    hole=0.3,
                                    marker_colors=['#2ecc71', '#e74c3c'],
                                    name="Testing"), 1, 2)
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature selection guidance
        st.markdown("### 🎯 Feature Selection Recommendations")
        
        feature_tabs = st.tabs(["High Correlation", "Low Variance", "Multicollinearity"])
        
        with feature_tabs[0]:
            st.markdown("""
            #### High Correlation with Target
            Features strongly correlated with churn (prioritize for modeling):
            
            **Keep:**
            - Tenure (usually negative correlation)
            - Contract type features
            - Internet service type
            - Payment method
            
            **Consider:**
            - Monthly charges
            - Total charges
            - Service add-ons
            """)
        
        with feature_tabs[1]:
            st.markdown("""
            #### Low Variance Features
            Features with little variation (consider removing):
            
            **Check for:**
            - Constant values (variance = 0)
            - Binary features with >95% one value
            - Features with missing values >50%
            
            **Impact:**
            - Little predictive power
            - Can cause overfitting
            - Increase model complexity
            """)
        
        with feature_tabs[2]:
            st.markdown("""
            #### Multicollinearity
            Highly correlated features (remove redundant ones):
            
            **Potential Issues:**
            - MonthlyCharges vs TotalCharges
            - Related service features
            - Derived features
            
            **Solution:**
            - Use Variance Inflation Factor (VIF)
            - Remove one from correlated pairs
            - Use dimensionality reduction
            """)
        
        # Model suggestions
        st.markdown("### 🤖 Recommended Models")
        
        model_col1, model_col2, model_col3 = st.columns(3)
        
        with model_col1:
            st.markdown("""
            #### 🏆 Top Performers
            **Logistic Regression**
            - Good baseline
            - Interpretable
            - Fast training
            
            **Random Forest**
            - Handles non-linearity
            - Robust to outliers
            - Feature importance
            
            **XGBoost**
            - State-of-the-art
            - Handles imbalance
            - High accuracy
            """)
        
        with model_col2:
            st.markdown("""
            #### 📊 For Comparison
            **Decision Tree**
            - Easy to interpret
            - Visualizable
            
            **SVM**
            - Good with many features
            - Effective in high-dim
            
            **Neural Network**
            - Complex patterns
            - Requires more data
            """)
        
        with model_col3:
            st.markdown("""
            #### 🎯 Evaluation Metrics
            
            **Primary:**
            - ROC-AUC
            - F1-Score
            - Precision-Recall
            
            **Secondary:**
            - Accuracy
            - Confusion Matrix
            - Business Metrics
            
            **For Imbalanced Data:**
            - Balanced Accuracy
            - Matthews Correlation
            - G-Mean
            """)
        
        # Export options
        st.markdown("### 📤 Export for Modeling")
        
        if st.session_state.df_processed is not None:
            csv = st.session_state.df_processed.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Processed Dataset",
                data=csv,
                file_name="telco_churn_model_ready.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p>📊 <strong>Telco Customer Churn Analytics Dashboard</strong> | Built with Streamlit</p>
    <p>For business inquiries: analytics@company.com | Last updated: 2024</p>
</div>
""", unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
from statsmodels.stats.proportion import proportions_ztest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Page configuration
st.set_page_config(
    page_title="E-Commerce Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load all datasets"""
    users = pd.read_csv("Data/users.csv", parse_dates=["signup_date"])
    transactions = pd.read_csv("Data/transactions.csv", parse_dates=["transaction_date"])
    retention = pd.read_csv("Data/retention.csv")
    return users, transactions, retention

# Initialize DuckDB connection and register tables
@st.cache_resource
def init_duckdb(users, transactions, retention):
    """Initialize DuckDB connection and register dataframes"""
    con = duckdb.connect(database=':memory:')
    con.register("users", users)
    con.register("transactions", transactions)
    con.register("retention", retention)
    return con

# Analysis functions
@st.cache_data
def get_funnel_overall(_con):
    """Get overall funnel metrics"""
    return _con.execute("""
    SELECT
       COUNT(*) AS signups,
       SUM(CASE WHEN U.activated = 'True' THEN 1 ELSE 0 END) AS activated_users,
       COUNT(DISTINCT(T.user_id)) AS purchasers,
       ROUND(SUM(CASE WHEN U.activated = 'True' THEN 1 ELSE 0 END)/COUNT(*), 4) AS activation_rate,
       ROUND(COUNT(DISTINCT(T.user_id))/SUM(CASE WHEN U.activated = 'True' THEN 1 ELSE 0 END), 4) AS purchase_rate_from_activated,
       ROUND(COUNT(DISTINCT(T.user_id))/COUNT(*), 4) AS overall_conversion
    FROM users U
    LEFT JOIN transactions T ON U.user_id = T.user_id
    """).df()

@st.cache_data
def get_funnel_by_dimension(_con, dimension):
    """Get funnel metrics by dimension"""
    query = f"""
    SELECT
       U.{dimension},
       COUNT(*) AS signups,
       SUM(CASE WHEN U.activated = 'True' THEN 1 ELSE 0 END) AS activated_users,
       COUNT(DISTINCT(T.user_id)) AS purchasers,
       ROUND(SUM(CASE WHEN U.activated = 'True' THEN 1 ELSE 0 END)/COUNT(*), 4) AS activation_rate,
       ROUND(COUNT(DISTINCT(T.user_id))/SUM(CASE WHEN U.activated = 'True' THEN 1 ELSE 0 END), 4) AS purchase_rate_from_activated,
       ROUND(COUNT(DISTINCT(T.user_id))/COUNT(*), 4) AS overall_conversion
    FROM Users U
    LEFT JOIN Transactions T ON U.user_id = T.user_id
    GROUP BY U.{dimension}
    ORDER BY overall_conversion DESC
    """
    return _con.execute(query).df()

@st.cache_data
def get_ab_test_results(_con, users):
    """Get A/B test results"""
    ab_summary = _con.execute("""
        SELECT
            experiment_group,
            COUNT(*) AS users,
            SUM(CASE WHEN activated THEN 1 ELSE 0 END) AS activated_users,
            COUNT(DISTINCT t.user_id) AS purchasers,
            ROUND(SUM(CASE WHEN activated THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS activation_rate,
            ROUND(COUNT(DISTINCT t.user_id) * 1.0 / COUNT(*), 4) AS overall_conversion
        FROM users u
        LEFT JOIN transactions t ON u.user_id = t.user_id
        GROUP BY experiment_group
    """).df()
    
    # Statistical test
    control = users[users["experiment_group"] == "control"]
    treatment = users[users["experiment_group"] == "treatment"]
    
    successes = np.array([
        control["activated"].sum(),
        treatment["activated"].sum()
    ])
    
    trials = np.array([
        len(control),
        len(treatment)
    ])
    
    stat, pval = proportions_ztest(successes, trials)
    
    control_rate = control["activated"].mean()
    treatment_rate = treatment["activated"].mean()
    lift = treatment_rate - control_rate
    relative_lift = lift / control_rate
    
    return ab_summary, pval, control_rate, treatment_rate, lift, relative_lift

@st.cache_data
def get_retention_data(retention, users):
    """Get retention analysis data"""
    retention_enriched = retention.merge(
        users[["user_id", "signup_date", "experiment_group", 
               "acquisition_channel", "device_type", "user_segment"]],
        on="user_id",
        how="left"
    )
    
    retention_enriched["signup_month"] = (
        retention_enriched["signup_date"]
        .dt.to_period("M")
        .astype(str)
    )
    
    overall_retention = (
        retention_enriched
        .groupby("week")["is_active"]
        .mean()
        .reset_index()
        .rename(columns={"is_active": "retention_rate"})
    )
    
    retention_by_experiment = (
        retention_enriched
        .groupby(["experiment_group", "week"])["is_active"]
        .mean()
        .reset_index()
        .rename(columns={"is_active": "retention_rate"})
    )
    
    cohort_table = (
        retention_enriched
        .groupby(["signup_month", "week"])["is_active"]
        .mean()
        .reset_index()
    )
    
    segment_retention = (
        retention_enriched
        .groupby(["user_segment", "week"])["is_active"]
        .mean()
        .reset_index()
    )
    
    return overall_retention, retention_by_experiment, cohort_table, segment_retention

@st.cache_data
def get_revenue_data(transactions, users):
    """Get revenue analysis data"""
    user_revenue = (
        transactions
        .groupby("user_id")["revenue"]
        .sum()
        .reset_index()
        .rename(columns={"revenue": "total_revenue"})
    )
    
    users_revenue = users.merge(
        user_revenue,
        on="user_id",
        how="left"
    )
    
    users_revenue["total_revenue"] = users_revenue["total_revenue"].fillna(0)
    
    users_revenue["signup_month"] = (
        users_revenue["signup_date"]
        .dt.to_period("M")
        .astype(str)
    )
    
    # Overall metrics
    total_revenue = users_revenue["total_revenue"].sum()
    arpu = users_revenue["total_revenue"].mean()
    paying_arpu = users_revenue.query("total_revenue > 0")["total_revenue"].mean()
    
    # Revenue by experiment
    revenue_by_experiment = (
        users_revenue
        .groupby("experiment_group")
        .agg(
            users=("user_id", "count"),
            total_revenue=("total_revenue", "sum"),
            arpu=("total_revenue", "mean"),
            paying_users=("total_revenue", lambda x: (x > 0).sum())
        )
        .reset_index()
    )
    
    # Cohort revenue
    cohort_revenue = (
        users_revenue
        .groupby("signup_month")
        .agg(
            users=("user_id", "count"),
            total_revenue=("total_revenue", "sum"),
            arpu=("total_revenue", "mean")
        )
        .reset_index()
        .sort_values("signup_month")
    )
    
    # Segment revenue
    segment_revenue = (
        users_revenue
        .groupby("user_segment")
        .agg(
            users=("user_id", "count"),
            total_revenue=("total_revenue", "sum"),
            arpu=("total_revenue", "mean")
        )
        .reset_index()
        .sort_values("arpu", ascending=False)
    )
    
    return (total_revenue, arpu, paying_arpu, revenue_by_experiment, 
            cohort_revenue, segment_revenue, users_revenue)

@st.cache_data
def get_churn_model(retention, users):
    """Build and evaluate churn prediction model"""
    final_week = 8
    
    churn_label = (
        retention
        .query(f"week == {final_week}")
        .rename(columns={"is_active": "is_active_final"})
        [["user_id", "is_active_final"]]
    )
    
    churn_label["churned"] = 1 - churn_label["is_active_final"]
    
    early_activity = (
        retention
        .query("week <= 4")
        .groupby("user_id")["is_active"]
        .mean()
        .reset_index()
        .rename(columns={"is_active": "early_activity_rate"})
    )
    
    features = (
        users
        .merge(early_activity, on="user_id", how="left")
        .merge(churn_label[["user_id", "churned"]], on="user_id", how="left")
    )
    
    features["early_activity_rate"] = features["early_activity_rate"].fillna(0)
    
    model_df = features[[
        "churned",
        "activated",
        "early_activity_rate",
        "experiment_group",
        "device_type",
        "user_segment"
    ]]
    
    model_df = pd.get_dummies(
        model_df,
        columns=["experiment_group", "device_type", "user_segment"],
        drop_first=True
    )
    
    X = model_df.drop("churned", axis=1)
    y = model_df["churned"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)
    
    coefficients = (
        pd.DataFrame({
            "feature": X.columns,
            "coefficient": model.coef_[0]
        })
        .sort_values("coefficient", ascending=False)
    )
    
    return auc, coefficients

# Main app
def main():
    st.markdown('<h1 class="main-header">📊 E-Commerce Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        users, transactions, retention = load_data()
        con = init_duckdb(users, transactions, retention)
    
    # Sidebar
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["Overview", "Funnel Analysis", "A/B Testing", "Retention Analysis", 
         "Revenue Analysis", "Churn Prediction"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data Summary**")
    st.sidebar.metric("Total Users", f"{len(users):,}")
    st.sidebar.metric("Total Transactions", f"{len(transactions):,}")
    st.sidebar.metric("Total Revenue", f"${transactions['revenue'].sum():,.2f}")
    
    # Page routing
    if page == "Overview":
        show_overview(users, transactions, retention, con)
    elif page == "Funnel Analysis":
        show_funnel_analysis(con)
    elif page == "A/B Testing":
        show_ab_testing(con, users)
    elif page == "Retention Analysis":
        show_retention_analysis(retention, users)
    elif page == "Revenue Analysis":
        show_revenue_analysis(transactions, users)
    elif page == "Churn Prediction":
        show_churn_prediction(retention, users)

def show_overview(users, transactions, retention, con):
    """Show overview dashboard"""
    st.header("📈 Business Overview")
    
    # Key metrics
    funnel = get_funnel_overall(con)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signups", f"{funnel['signups'].values[0]:,}")
    with col2:
        st.metric("Activated Users", f"{funnel['activated_users'].values[0]:,}",
                 f"{funnel['activation_rate'].values[0]*100:.1f}%")
    with col3:
        st.metric("Purchasers", f"{funnel['purchasers'].values[0]:,}",
                 f"{funnel['overall_conversion'].values[0]*100:.1f}%")
    with col4:
        total_rev = transactions['revenue'].sum()
        st.metric("Total Revenue", f"${total_rev:,.0f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Funnel visualization
        st.subheader("Conversion Funnel")
        funnel_data = pd.DataFrame({
            'Stage': ['Signups', 'Activated', 'Purchasers'],
            'Count': [
                funnel['signups'].values[0],
                funnel['activated_users'].values[0],
                funnel['purchasers'].values[0]
            ]
        })
        
        fig = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Count'],
            textinfo="value+percent previous",
            marker=dict(color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # User distribution
        st.subheader("User Segments")
        segment_dist = users['user_segment'].value_counts()
        fig = px.pie(
            values=segment_dist.values,
            names=segment_dist.index,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Acquisition Channels")
        channel_dist = users['acquisition_channel'].value_counts()
        fig = px.bar(
            x=channel_dist.index,
            y=channel_dist.values,
            labels={'x': 'Channel', 'y': 'Users'},
            color=channel_dist.index
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Device Type Distribution")
        device_dist = users['device_type'].value_counts()
        fig = px.bar(
            x=device_dist.index,
            y=device_dist.values,
            labels={'x': 'Device', 'y': 'Users'},
            color=device_dist.index
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def show_funnel_analysis(con):
    """Show funnel analysis"""
    st.header("🎯 Funnel Analysis")
    
    # Overall funnel
    st.subheader("Overall Conversion Funnel")
    funnel_overall = get_funnel_overall(con)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Activation Rate", f"{funnel_overall['activation_rate'].values[0]*100:.2f}%")
    with col2:
        st.metric("Purchase Rate (from Activated)", 
                 f"{funnel_overall['purchase_rate_from_activated'].values[0]*100:.2f}%")
    with col3:
        st.metric("Overall Conversion", f"{funnel_overall['overall_conversion'].values[0]*100:.2f}%")
    
    st.markdown("---")
    
    # Dimension selector
    dimension = st.selectbox(
        "Analyze by:",
        ["device_type", "acquisition_channel", "user_segment", "country"]
    )
    
    funnel_by_dim = get_funnel_by_dimension(con, dimension)
    
    # Display table
    st.subheader(f"Funnel by {dimension.replace('_', ' ').title()}")
    st.dataframe(funnel_by_dim, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Activation Rate by " + dimension.replace('_', ' ').title())
        fig = px.bar(
            funnel_by_dim,
            x=dimension,
            y='activation_rate',
            color='activation_rate',
            color_continuous_scale='Blues',
            labels={dimension: dimension.replace('_', ' ').title(), 
                   'activation_rate': 'Activation Rate'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Overall Conversion by " + dimension.replace('_', ' ').title())
        fig = px.bar(
            funnel_by_dim,
            x=dimension,
            y='overall_conversion',
            color='overall_conversion',
            color_continuous_scale='Greens',
            labels={dimension: dimension.replace('_', ' ').title(), 
                   'overall_conversion': 'Overall Conversion'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Volume vs Conversion scatter
    st.subheader("Volume vs. Conversion Rate")
    fig = px.scatter(
        funnel_by_dim,
        x='signups',
        y='overall_conversion',
        size='purchasers',
        color=dimension,
        hover_data=[dimension, 'signups', 'activated_users', 'purchasers'],
        labels={'signups': 'Total Signups', 'overall_conversion': 'Overall Conversion Rate'}
    )
    st.plotly_chart(fig, use_container_width=True)

def show_ab_testing(con, users):
    """Show A/B test results"""
    st.header("🧪 A/B Testing Analysis")
    
    ab_summary, pval, control_rate, treatment_rate, lift, relative_lift = get_ab_test_results(con, users)
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Control Activation Rate", f"{control_rate*100:.2f}%")
    with col2:
        st.metric("Treatment Activation Rate", f"{treatment_rate*100:.2f}%")
    with col3:
        st.metric("Absolute Lift", f"{lift*100:.2f}%")
    with col4:
        st.metric("Relative Lift", f"{relative_lift*100:.2f}%")
    
    # Statistical significance
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Statistical Significance")
        if pval < 0.05:
            st.success(f"✅ **Statistically Significant** (p-value: {pval:.4f})")
            st.write("The difference between control and treatment is statistically significant at α = 0.05")
        else:
            st.warning(f"⚠️ **Not Statistically Significant** (p-value: {pval:.4f})")
            st.write("The difference between control and treatment is not statistically significant at α = 0.05")
    
    with col2:
        st.metric("P-value", f"{pval:.4f}")
        st.metric("Significance Level (α)", "0.05")
    
    st.markdown("---")
    
    # Detailed comparison
    st.subheader("Detailed Group Comparison")
    st.dataframe(ab_summary, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Activation Rate Comparison")
        fig = px.bar(
            ab_summary,
            x='experiment_group',
            y='activation_rate',
            color='experiment_group',
            labels={'experiment_group': 'Group', 'activation_rate': 'Activation Rate'},
            text='activation_rate'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Overall Conversion Comparison")
        fig = px.bar(
            ab_summary,
            x='experiment_group',
            y='overall_conversion',
            color='experiment_group',
            labels={'experiment_group': 'Group', 'overall_conversion': 'Overall Conversion'},
            text='overall_conversion'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment-level effects
    st.subheader("Segment-Level Effects")
    segment_effect = con.execute("""
        SELECT
            device_type,
            experiment_group,
            COUNT(*) AS users,
            ROUND(SUM(CASE WHEN activated THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS activation_rate
        FROM users
        GROUP BY device_type, experiment_group
        ORDER BY device_type, experiment_group
    """).df()
    
    fig = px.bar(
        segment_effect,
        x='device_type',
        y='activation_rate',
        color='experiment_group',
        barmode='group',
        labels={'device_type': 'Device Type', 'activation_rate': 'Activation Rate'},
        text='activation_rate'
    )
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

def show_retention_analysis(retention, users):
    """Show retention analysis"""
    st.header("📅 Retention Analysis")
    
    overall_ret, retention_by_exp, cohort_table, segment_ret = get_retention_data(retention, users)
    
    # Overall retention curve
    st.subheader("Overall Retention Curve")
    fig = px.line(
        overall_ret,
        x='week',
        y='retention_rate',
        markers=True,
        labels={'week': 'Week', 'retention_rate': 'Retention Rate'}
    )
    fig.update_traces(line_color='#1f77b4', line_width=3)
    fig.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig, use_container_width=True)
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        week1_ret = overall_ret[overall_ret['week'] == 1]['retention_rate'].values[0]
        st.metric("Week 1 Retention", f"{week1_ret*100:.1f}%")
    with col2:
        week4_ret = overall_ret[overall_ret['week'] == 4]['retention_rate'].values[0]
        st.metric("Week 4 Retention", f"{week4_ret*100:.1f}%")
    with col3:
        week8_ret = overall_ret[overall_ret['week'] == 8]['retention_rate'].values[0]
        st.metric("Week 8 Retention", f"{week8_ret*100:.1f}%")
    
    st.markdown("---")
    
    # Retention by experiment group
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Retention by Experiment Group")
        fig = px.line(
            retention_by_exp,
            x='week',
            y='retention_rate',
            color='experiment_group',
            markers=True,
            labels={'week': 'Week', 'retention_rate': 'Retention Rate', 
                   'experiment_group': 'Experiment Group'}
        )
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Retention by User Segment")
        fig = px.line(
            segment_ret,
            x='week',
            y='is_active',
            color='user_segment',
            markers=True,
            labels={'week': 'Week', 'is_active': 'Retention Rate', 
                   'user_segment': 'User Segment'}
        )
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cohort analysis
    st.subheader("Cohort Retention Heatmap")
    cohort_pivot = cohort_table.pivot(
        index='signup_month',
        columns='week',
        values='is_active'
    )
    
    fig = px.imshow(
        cohort_pivot,
        labels=dict(x="Week", y="Signup Month", color="Retention Rate"),
        color_continuous_scale='RdYlGn',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

def show_revenue_analysis(transactions, users):
    """Show revenue analysis"""
    st.header("💰 Revenue Analysis")
    
    (total_revenue, arpu, paying_arpu, revenue_by_experiment,
     cohort_revenue, segment_revenue, users_revenue) = get_revenue_data(transactions, users)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    with col2:
        st.metric("ARPU", f"${arpu:.2f}")
    with col3:
        st.metric("Paying ARPU", f"${paying_arpu:.2f}")
    with col4:
        paying_users = (users_revenue['total_revenue'] > 0).sum()
        conversion_rate = paying_users / len(users_revenue)
        st.metric("Paying Users %", f"{conversion_rate*100:.1f}%")
    
    st.markdown("---")
    
    # Revenue by experiment
    st.subheader("Revenue by Experiment Group")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(revenue_by_experiment, use_container_width=True)
    
    with col2:
        fig = px.bar(
            revenue_by_experiment,
            x='experiment_group',
            y='arpu',
            color='experiment_group',
            labels={'experiment_group': 'Experiment Group', 'arpu': 'ARPU ($)'},
            text='arpu'
        )
        fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cohort revenue
    st.subheader("Revenue by Signup Cohort")
    fig = px.line(
        cohort_revenue,
        x='signup_month',
        y='arpu',
        markers=True,
        labels={'signup_month': 'Signup Month', 'arpu': 'ARPU ($)'}
    )
    fig.update_traces(line_color='#2ca02c', line_width=3)
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment revenue
    st.subheader("Revenue by User Segment")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(segment_revenue, use_container_width=True)
    
    with col2:
        fig = px.bar(
            segment_revenue,
            x='user_segment',
            y='arpu',
            color='arpu',
            color_continuous_scale='Viridis',
            labels={'user_segment': 'User Segment', 'arpu': 'ARPU ($)'},
            text='arpu'
        )
        fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
    
    # Revenue distribution
    st.subheader("Revenue Distribution")
    fig = px.histogram(
        users_revenue[users_revenue['total_revenue'] > 0],
        x='total_revenue',
        nbins=50,
        labels={'total_revenue': 'Total Revenue ($)'},
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def show_churn_prediction(retention, users):
    """Show churn prediction analysis"""
    st.header("🔮 Churn Prediction")
    
    with st.spinner("Training churn prediction model..."):
        auc, coefficients = get_churn_model(retention, users)
    
    # Model performance
    st.subheader("Model Performance")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("ROC-AUC Score", f"{auc:.4f}")
        
        if auc >= 0.8:
            st.success("✅ Excellent model performance")
        elif auc >= 0.7:
            st.info("ℹ️ Good model performance")
        else:
            st.warning("⚠️ Fair model performance")
    
    with col2:
        st.write("**Model Interpretation:**")
        st.write(f"- ROC-AUC Score: {auc:.4f}")
        st.write("- This score indicates how well the model can distinguish between users who will churn and those who won't")
        st.write("- A score of 0.5 is random, and 1.0 is perfect")
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("Feature Importance (Coefficients)")
    
    # Split into positive and negative
    positive_coef = coefficients[coefficients['coefficient'] > 0].head(10)
    negative_coef = coefficients[coefficients['coefficient'] < 0].head(10)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Factors Increasing Churn Risk** (Positive Coefficients)")
        fig = px.bar(
            positive_coef,
            y='feature',
            x='coefficient',
            orientation='h',
            labels={'feature': 'Feature', 'coefficient': 'Coefficient'},
            color='coefficient',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Factors Decreasing Churn Risk** (Negative Coefficients)")
        fig = px.bar(
            negative_coef,
            y='feature',
            x='coefficient',
            orientation='h',
            labels={'feature': 'Feature', 'coefficient': 'Coefficient'},
            color='coefficient',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Full coefficient table
    with st.expander("View All Feature Coefficients"):
        st.dataframe(coefficients, use_container_width=True)
    
    # Insights
    st.markdown("---")
    st.subheader("Key Insights")
    
    top_protective = coefficients.iloc[-1]['feature']
    top_risk = coefficients.iloc[0]['feature']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Most Protective Factor**: {top_protective}")
        st.write("Users with this characteristic have the lowest churn risk")
    
    with col2:
        st.warning(f"**Highest Risk Factor**: {top_risk}")
        st.write("Users with this characteristic have the highest churn risk")

if __name__ == "__main__":
    main()
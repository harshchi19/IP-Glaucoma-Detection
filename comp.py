# comparison.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.title("🔍 Glaucoma Detection Model Comparison Dashboard")

    # Create DataFrame for model results
    results = [
    {'Model name': 'SVM', 'Accuracy': 0.7535211267605634, 'Precision': 0.7288135593220338, 'Recall': 0.6935483870967742, 'F1 score': 0.7107438016528926},
    {'Model name': 'Random Forest', 'Accuracy': 0.7112676056338029, 'Precision': 0.6615384615384615, 'Recall': 0.6935483870967742, 'F1 score': 0.6771653543307087},
    {'Model name': 'Logistic Regression', 'Accuracy': 0.7605633802816901, 'Precision': 0.7, 'Recall': 0.7903225806451613, 'F1 score': 0.7424242424242423},
    {'Model name': 'Naive Bayes Gaussian', 'Accuracy': 0.5211267605633803, 'Precision': 0.4634146341463415, 'Recall': 0.6129032258064516, 'F1 score': 0.5277777777777778},
    {'Model name': 'Naive Bayes Multinomial', 'Accuracy': 0.676056338028169, 'Precision': 0.6333333333333333, 'Recall': 0.6129032258064516, 'F1 score': 0.6229508196721313},
    {'Model name': 'Decision Tree', 'Accuracy': 0.6056338028169014, 'Precision': 0.5441176470588235, 'Recall': 0.5967741935483871, 'F1 score': 0.5692307692307692},
    {'Model name': 'Adaboost with DecisionTree', 'Accuracy': 0.5845070422535211, 'Precision': 0.5254237288135594, 'Recall': 0.5, 'F1 score': 0.5123966942148761},
    {'Model name': 'Adaboost with LogisticRegrission', 'Accuracy': 0.7746478873239436, 'Precision': 0.7205882352941176, 'Recall': 0.7903225806451613, 'F1 score': 0.7538461538461538},
    {'Model name': 'Gradient Boost', 'Accuracy': 0.5704225352112676, 'Precision': 0.5079365079365079, 'Recall': 0.5161290322580645, 'F1 score': 0.512},
    {'Model name': 'CNN', 'Accuracy': 0.92842149, 'Precision': 0.913793, 'Recall': 0.854839, 'F1 score': 0.883333},
    {'Model name': 'ResNet-18', 'Accuracy': 0.9983, 'Precision': 0.9975, 'Recall': 0.9970, 'F1 score': 0.9972},
    {'Model name': 'XGBoost', 'Accuracy': 0.95, 'Precision': 0.9456, 'Recall': 0.9489, 'F1 score': 0.9472},
    {'Model name': 'Custom Model', 'Accuracy': 0.99, 'Precision': 0.9889, 'Recall': 0.9892, 'F1 score': 0.9890}
    ]
    df_results = pd.DataFrame(results)

    # Enhanced dataset composition with access links
    composition_data = {
    'Dataset': [
        'BEH (Bangladesh Eye Hospital)', 'CRFO-v4', 'DR-HAGIS', 'DRISHTI-GS1-TRAIN',
        'DRISHTI-GS1-TEST', 'EyePACS-AIROGS', 'FIVES', 'G1020', 'HRF', 'JSIEC-1000',
        'LES-AV', 'OIA-ODIR-TRAIN', 'OIA-ODIR-TEST-ONLINE', 'OIA-ODIR-TEST-OFFLINE',
        'ORIGA-light', 'PAPILA', 'REFUGE1-TRAIN', 'REFUGE1-VALIDATION', 'sjchoi86-HRF'
    ],
    '0': [463, 31, 0, 18, 13, 0, 200, 724, 15, 38, 11, 2932, 802, 417, 482, 333, 360, 360, 300],
    '1': [171, 48, 10, 32, 38, 3269, 200, 296, 15, 0, 11, 197, 58, 36, 168, 87, 40, 40, 101],
    '-1': [0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 0, 18, 25, 9, 0, 68, 0, 0, 0],
    'Access Link': [
        'https://github.com/mirtanvirislam/Deep-Learning-Based-Glaucoma-Detection-with-Cropped-Optic-Cup-and-Disc-and-Blood-Vessel-Segmentation/tree/master/Dataset',
        'https://data.mendeley.com/datasets/trghs22fpg/4',
        'https://personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/',
        'https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php',
        'https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php',
        'https://airogs.grand-challenge.org/data-and-challenge/',
        'https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1',
        'https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets',
        'https://www5.cs.fau.de/research/data/fundus-images/',
        'https://www.kaggle.com/datasets/linchundan/fundusimage1000',
        'https://figshare.com/articles/dataset/LES-AV_dataset/11857698/1',
        'https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k',
        'https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k',
        'https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k',
        'https://www.kaggle.com/datasets/sshikamaru/glaucoma-detection',
        'https://doi.org/10.6084/m9.figshare.14798004.v1',
        'https://refuge.grand-challenge.org/REFUGE2Download/',
        'https://refuge.grand-challenge.org/REFUGE2Download/',
        'https://github.com/yiweichen04/retina_dataset'
    ]
    }
    df_composition = pd.DataFrame(composition_data)
    
    totals = {
    'Dataset': 'Total',
    '0': df_composition['0'].sum(),
    '1': df_composition['1'].sum(),
    '-1': df_composition['-1'].sum(),
    'Access Link': ''
    }
    df_composition = pd.concat([df_composition, pd.DataFrame([totals])], ignore_index=True)

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📊 Model Comparison", "📈 Detailed Metrics", "📋 Dataset Information"])

    with tab1:
        st.header("Model Performance Comparison")
        
        # Create a bar chart for all metrics
        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=df_results['Model name'],
                y=df_results[metric],
                text=df_results[metric].round(3),
                textposition='auto',
                marker_color=colors[i]
            ))

        fig.update_layout(
            barmode='group',
            title="Model Performance Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=600,
            xaxis_tickangle=-45,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Display the results table with conditional formatting
        st.subheader("Detailed Results Table")
        
        def highlight_best_metric(s):
            is_max = s == s.max()
            return ['background-color: #90EE90' if v else '' for v in is_max]
        
        styled_df = df_results.style.format({
            'Accuracy': '{:.3f}',
            'Precision': '{:.3f}',
            'Recall': '{:.3f}',
            'F1 score': '{:.3f}'
        }).apply(highlight_best_metric, subset=['Accuracy', 'Precision', 'Recall', 'F1 score'])
        
        st.dataframe(styled_df)

    with tab2:
        st.header("Detailed Metric Analysis")
        
        # Metric selector
        metric = st.selectbox(
            "Select Metric to Visualize",
            ['Accuracy', 'Precision', 'Recall', 'F1 score']
        )
        
        # Create sorted bar chart for selected metric
        fig = px.bar(
            df_results.sort_values(metric, ascending=False),
            x='Model name',
            y=metric,
            title=f"{metric} Comparison Across Models",
            text=df_results[metric].round(3),
            color=metric,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title=metric,
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show best and worst performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 3 Performers")
            top_3 = df_results.nlargest(3, metric)[['Model name', metric]]
            st.dataframe(top_3.style.format({metric: '{:.3f}'}).background_gradient(cmap='Greens'))
            
        with col2:
            st.subheader("Bottom 3 Performers")
            bottom_3 = df_results.nsmallest(3, metric)[['Model name', metric]]
            st.dataframe(bottom_3.style.format({metric: '{:.3f}'}).background_gradient(cmap='Reds'))

        # Add performance distribution
        st.subheader("Performance Distribution")
        fig = px.box(
            df_results,
            y=['Accuracy', 'Precision', 'Recall', 'F1 score'],
            title="Distribution of Performance Metrics Across All Models"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Dataset Information")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Dataset distribution pie chart
            total_samples = df_composition.iloc[:-1][['0', '1', '-1']].sum().sum()
            fig = px.pie(
                values=[df_composition.iloc[:-1]['0'].sum(), 
                    df_composition.iloc[:-1]['1'].sum(), 
                    df_composition.iloc[:-1]['-1'].sum()],
                names=['Non-Glaucoma', 'Glaucoma', 'Glaucoma-suspect'],
                title=f'Dataset Class Distribution (Total: {total_samples:,} images)',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Label Distribution")
            label_dist = pd.DataFrame({
                'Label': ['Non-Glaucoma', 'Glaucoma', 'Glaucoma-suspect'],
                'Code': [0, 1, -1],
                'Count': [df_composition.iloc[:-1]['0'].sum(), 
                        df_composition.iloc[:-1]['1'].sum(), 
                        df_composition.iloc[:-1]['-1'].sum()]
            })
            st.dataframe(label_dist.style.background_gradient(subset=['Count'], cmap='Blues'))
        
        st.markdown("""
        ### Dataset Composition
        The SMDG-19 dataset is comprised of the following public glaucoma image datasets. 
        Click on the links to access each dataset:
        """)
        
        # Custom CSS to style the table
        st.markdown("""
        <style>
        .streamlit-expanderHeader {
            font-size: 16px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display the dataset composition with clickable links
        st.markdown("### Detailed Dataset Composition")
        
        # Convert DataFrame to HTML with clickable links
        def make_clickable(link):
            return f'<a href="{link}" target="_blank">Access Dataset</a>' if link else ''
        
        df_display = df_composition.copy()
        df_display['Access Link'] = df_display['Access Link'].apply(make_clickable)
        
        st.write(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        
        # Additional dataset information
        st.markdown("""
        ### Label Information
        - **0**: Non-Glaucoma instance
        - **1**: Glaucoma instance
        - **-1**: Glaucoma-suspect instance
        
        ### Dataset Usage Notes
        - All images are fundus photographs
        - Images come from diverse sources and equipment
        - Some datasets may require registration or approval for access
        - Please cite the original dataset sources when using them in research
        """)

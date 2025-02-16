import streamlit as st
import joblib
import pandas as pd

# 加载预训练模型和预处理器
model = joblib.load('random_forest_final_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('age_scaler.pkl')
features = joblib.load('features.pkl')

# 创建特征输入界面
st.title('甲状腺癌复发预测系统')
st.markdown('### 请输入患者特征参数')

# 定义各特征的选项字典
feature_options = {
    'Gender': ['F', 'M'],
    'Smoking': ['No', 'Yes'],
    'Hx Smoking': ['No', 'Yes'],
    'Hx Radiothreapy': ['No', 'Yes'],
    'Thyroid Function': ['Euthyroid', 'Clinical Hyperthyroidism', 
                         'Clinical Hypothyroidism', 'Subclinical Hyperthyroidism',
                         'Subclinical Hypothyroidism'],
    'Physical Examination': ['Single nodular goiter-left', 'Multinodular goiter',
                             'Single nodular goiter-right', 'Normal', 'Diffuse goiter'],
    'Adenopathy': ['No', 'Right', 'Extensive', 'Left', 'Bilateral', 'Posterior'],
    'Pathology': ['Micropapillary', 'Papillary', 'Follicular', 'Hurthel cell'],
    'Focality': ['Uni-Focal', 'Multi-Focal'],
    'Risk': ['Low', 'Intermediate', 'High'],
    'T': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'],
    'N': ['N0', 'N1b', 'N1a'],
    'M': ['M0', 'M1'],
    'Stage': ['I', 'II', 'IVB', 'III', 'IVA'],
    'Response': ['Indeterminate', 'Excellent', 
                 'Structural Incomplete', 'Biochemical Incomplete']
}

# 创建输入控件
input_data = {}
cols = st.columns(3)
for i, feature in enumerate(features):
    if feature == 'Age':
        input_data[feature] = cols[i % 3].number_input(
            'Age', min_value=0, max_value=100, value=45
        )
    else:
        input_data[feature] = cols[i % 3].selectbox(
            feature, 
            options=feature_options[feature]
        )

# 创建预测按钮
if st.button('开始预测'):
    # 转换输入数据
    processed_data = {}
    for feature in features:
        if feature == 'Age':
            # 标准化年龄
            processed_data[feature] = scaler.transform([[input_data[feature]]])[0][0]
        else:
            # 对分类特征进行编码
            le = label_encoders[feature]
            processed_data[feature] = le.transform([input_data[feature]])[0]
    
    # 转换为DataFrame并保持特征顺序
    input_df = pd.DataFrame([processed_data], columns=features)
    
    # 进行预测
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]
    
    # 显示结果
    st.success('### 预测结果：')
    result = '可能复发（Yes）' if prediction[0] == 1 else '未复发（No）'
    st.markdown(f"**预测结论**: {result}")
    st.markdown(f"**置信度**: \
        No: {probability[0]*100:.1f}% | Yes: {probability[1]*100:.1f}%")


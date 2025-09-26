
#! Author :Wei Kai


import time
import pandas as pd
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import os
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error



if __name__=="__main__":
    start=time.time()
    # region open Excel file and read datas
    path=r"data.xlsx"  # file address
    # "D:\pressure\data.xlsx"
    #df_WellT = pd.read_excel(path,sheet_name=u'DB102') #type: DataFrame
    #df_WellA = pd.read_excel(path,sheet_name=u'DB1') 
    #df_WellB = pd.read_excel(path,sheet_name=u'DB2') 
    #df_WellC = pd.read_excel(path,sheet_name=u'DB101') 
    #df_WellD = pd.read_excel(path,sheet_name=u'DB103') 

    df= pd.read_excel(path,sheet_name=None) #read all sheet, improve efficiency
    #df_WellT= df.sheets()[0]
    df_WellT=df['DB102'] 
    df_WellA=df['DB1']; 
    df_WellB=df['DB2'] ; 
    df_WellC=df['DB101'];   
    df_WellD=df['DB103']
        
    #welldepth
    HT=df_WellT.iloc[0:7737,0].values  
    HA=df_WellA.iloc[:,0].values
    HB=df_WellB.iloc[:,0].values
    HC=df_WellC.iloc[:,0].values
    HD=df_WellD.iloc[:,0].values

    #geological parameters: porepressure
    pT=df_WellT.iloc[0:7737,1].values+0.25
    pA=df_WellA.iloc[:,1].values
    pB=df_WellB.iloc[:,1].values
    pC=df_WellC.iloc[:,1].values
    pD=df_WellD.iloc[:,1].values
    #endregion


    hT_f11=3120;hT_f12=4273; 
    hA_f11=3750;  hA_f12=4350;
    hB_f11=3895;  hB_f12=4209;
    hC_f11=3716;  hC_f12=4684;
    hD_f11=3522;  hD_f12=4750;

    
    t=1;
    thT=HT[2]-HT[1];
    thA=HA[2]-HA[1];
    thB=HB[2]-HB[1];
    thC=HC[2]-HC[1];
    thD=HD[2]-HD[1];
    
   
    nT1=int((hT_f11-HT[1])/(thT*t)+2); nT2=int((hT_f12-HT[1])/(thT*t)+2);
    nA1=int((hA_f11-HA[1])/(thA*t)+2); nA2=int((hA_f12-HA[1])/(thA*t)+2);
    nB1=int((hB_f11-HB[1])/(thB*t)+2); nB2=int((hB_f12-HB[1])/(thB*t)+2);
    nC1=int((hC_f11-HC[1])/(thC*t)+2); nC2=int((hC_f12-HC[1])/(thC*t)+2);
    nD1=int((hD_f11-HD[1])/(thD*t)+2); nD2=int((hD_f12-HD[1])/(thD*t)+2);
    
    
    hT_f1=HT[nT1:nT2];
    hA_f1=HA[nA1:nA2];
    hB_f1=HB[nB1:nB2];
    hC_f1=HC[nC1:nC2];
    hD_f1=HD[nD1:nD2];
    
   
    pT_f1=pT[nT1:nT2];
    pA_f1=pA[nA1:nA2];
    pB_f1=pB[nB1:nB2];
    pC_f1=pC[nC1:nC2];
    pD_f1=pD[nD1:nD2]
    #endregion

   
    h_f1_TT=hT_f11*np.ones((len(hT_f1),1))+(hT_f1.reshape(-1,1)-HT[nT1]*np.ones((len(hT_f1),1)))*(hT_f12-hT_f11)/(HT[nT2]-HT[nT1]);   
    h_f1_TA=hT_f11*np.ones((len(hA_f1),1))+(hA_f1.reshape(-1,1)-HA[nA1]*np.ones((len(hA_f1),1)))*(hT_f12-hT_f11)/(HA[nA2]-HA[nA1]);   
    h_f1_TB=hT_f11*np.ones((len(hB_f1),1))+(hB_f1.reshape(-1,1)-HB[nB1]*np.ones((len(hB_f1),1)))*(hT_f12-hT_f11)/(HB[nB2]-HB[nB1]);   
    h_f1_TC=hT_f11*np.ones((len(hC_f1),1))+(hC_f1.reshape(-1,1)-HC[nC1]*np.ones((len(hC_f1),1)))*(hT_f12-hT_f11)/(HC[nC2]-HC[nC1]);   
    h_f1_TD=hT_f11*np.ones((len(hD_f1),1))+(hD_f1.reshape(-1,1)-HD[nD1]*np.ones((len(hD_f1),1)))*(hT_f12-hT_f11)/(HD[nD2]-HD[nD1]);  

   
    h_f1=np.arange(hT_f11,hT_f12,1)
    fT=interp1d(np.array(h_f1_TT).flatten(),np.array(pT_f1).flatten(),kind='linear',bounds_error = None,fill_value="extrapolate");    
    pT_f1_uni=fT(h_f1)
    fA=interp1d(np.array(h_f1_TA).flatten(),np.array(pA_f1).flatten(),kind='linear',bounds_error = None,fill_value="extrapolate");    
    pA_f1_uni=fA(h_f1)
    fB=interp1d(np.array(h_f1_TB).flatten(),np.array(pB_f1).flatten(),kind='linear',bounds_error = None,fill_value="extrapolate");    
    pB_f1_uni=fB(h_f1)
    fC=interp1d(np.array(h_f1_TC).flatten(),np.array(pC_f1).flatten(),kind='linear',bounds_error = None,fill_value="extrapolate");    
    pC_f1_uni=fC(h_f1)
    fD=interp1d(np.array(h_f1_TD).flatten(),np.array(pD_f1).flatten(),kind='linear',bounds_error = None,fill_value="extrapolate");    
    pD_f1_uni=fD(h_f1)
    

    
   
    XY=np.array([[3750,4350],
        [3895, 4209],
        [3716, 4684],
        [3522, 4750]])
    X0=XY[:,0];Y0=XY[:,1]
    
    X_T=[3120];Y_T=[4273];
   
    p_offset=np.array(list(zip(pA_f1_uni,pB_f1_uni,pC_f1_uni,pD_f1_uni)))
    pT_wa=np.zeros((len(h_f1)));

   
    d_TO=np.zeros((len(h_f1),XY.shape[0])) #d[h_id,well_id] distance between offset wells and target well
    p=2;  
    i=0
    while i<len(h_f1):
        j=0
        while j<XY.shape[0]:
            
            d_TO[i,j]=sqrt((X0[j]-X_T)**2+(Y0[j]-Y_T)**2)
            j+=1
       
        D=d_TO[i,0]**p+d_TO[i,1]**p+d_TO[i,2]**p+d_TO[i,3]**p
        pT_wa[i]=(d_TO[i,0]**p/D)*pA_f1_uni[i]+(d_TO[i,1]**p/D)*pB_f1_uni[i]+(d_TO[i,2]**p/D)*pC_f1_uni[i]+(d_TO[i,3]**p/D)*pD_f1_uni[i]
        i+=1

    pT_lower=np.minimum.reduce((pA_f1_uni,pB_f1_uni,pC_f1_uni,pD_f1_uni),axis=0)
    pT_upper=np.maximum.reduce((pA_f1_uni,pB_f1_uni,pC_f1_uni,pD_f1_uni),axis=0)
  
    

    
    min_length = min(len(pA_f1_uni), len(pB_f1_uni), len(pC_f1_uni), len(pD_f1_uni), len(pT_wa), len(pT_f1))

    
    pA_f1_uni = pA_f1_uni[:min_length]
    pB_f1_uni = pB_f1_uni[:min_length]
    pC_f1_uni = pC_f1_uni[:min_length]
    pD_f1_uni = pD_f1_uni[:min_length]
    pT_wa = pT_wa[:min_length]
    pT_f1_uni = pT_f1_uni[:min_length]
    
    
    X_all = np.column_stack([pA_f1_uni, pB_f1_uni, pC_f1_uni, pD_f1_uni]) 
    y_all = pT_wa 

   
    X_test = np.column_stack([pA_f1_uni, pB_f1_uni, pC_f1_uni, pD_f1_uni])  
    y_test = pT_f1_uni  


   
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    X_test_scaled = scaler.transform(X_test)

 
    X_train, X_val, y_train, y_val = train_test_split(X_all_scaled, y_all, test_size=0.2, random_state=42)


    
    xgb_model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        min_child_weight=10,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha=1, 
        reg_lambda=1,
        early_stopping_round=20
    )
    

  
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

  
    y_val_pred = xgb_model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_medar = median_absolute_error(y_val, y_val_pred)
    val_mar = np.mean(y_val - y_val_pred)
    val_mre = np.mean(np.abs((y_val - y_val_pred) / y_val))

    print(f"Validation MSE: {val_mse:.6f}")
    print(f"Validation R² Score: {val_r2:.4f}")
    print(f"Validation MAE: {val_mae:.6f}")
    print(f"Validation MedAR: {val_medar:.6f}")
    print(f"Validation MAR: {val_mar:.6f}")
    print(f"Validation MRE: {val_mre:.6f}")

   
    y_pred_test = xgb_model.predict(X_test_scaled)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_medar = median_absolute_error(y_test, y_pred_test)
    test_mar = np.mean(y_test - y_pred_test)
    test_mre = np.mean(np.abs((y_test - y_pred_test) / y_test))

    print(f"Test MSE (Target Well): {test_mse:.6f}")
    print(f"Test R² Score (Target Well): {test_r2:.4f}")
    print(f"Test MAE (Target Well): {test_mae:.6f}")
    print(f"Test MedAR (Target Well): {test_medar:.6f}")
    print(f"Test MAR (Target Well): {test_mar:.6f}")
    print(f"Test MRE (Target Well): {test_mre:.6f}")

  
    
    msee = mean_squared_error(pT_f1_uni, pT_wa)
    print(f"原始Mean Squared Error (MSE): {msee}")
    maae = mean_absolute_error(pT_f1_uni, pT_wa)
    print(f"原始MAE (Mean Absolute Error): {maae}")
    maar = np.mean(pT_f1_uni - pT_wa)
    print(f"原始MAR (Mean Absolute Residual): {pT_wa}")
   
    medaar = median_absolute_error(pT_f1_uni, pT_wa)
    print(f"原始MedAR (Median Absolute Residual): {medaar}")
   
    mrre = np.mean(np.abs((pT_f1_uni - pT_wa) / pT_f1_uni))
    print(f"原始MRE (Mean Relative Error): {mrre:.4f}")



  
    msse = mean_squared_error(pT_f1_uni, y_pred_test)
    print(f"结果Mean Squared Error (MSE): {msse}")
    maee = mean_absolute_error(pT_f1_uni, y_pred_test)
    print(f"结果MAE (Mean Absolute Error): {maee}")
    marr = np.mean(pT_f1_uni - y_pred_test)
    print(f"结果MAR (Mean Absolute Residual): {marr}")
   
    medarr = median_absolute_error(pT_f1_uni, y_pred_test)
    print(f"结果MedAR (Median Absolute Residual): {medarr}")
    
    mree = np.mean(np.abs((pT_f1_uni - y_pred_test) / pT_f1_uni))
    print(f"结果MRE (Mean Relative Error): {mree:.4f}")


 
    comparison_table = pd.DataFrame({
        "Depth (m)": h_f1,  
        "Actual (g/cm³)": pT_f1_uni,      
        "Predicted (g/cm³)": y_pred_test,  
        "Weighted Avg (g/cm³)": pT_wa 
    })


   
    ewput_file = "D:\\pressure\\ewput\\ewaa.xlsx"
    comparison_table.to_excel(ewput_file, index=False)
    print(f"Results saved to {ewput_file}")

    ewput_file_folder = "D:/pressure/ewput"
    os.makedirs(ewput_file_folder, exist_ok=True)

    
    depth = comparison_table["Depth (m)"]
    actual = comparison_table["Actual (g/cm³)"]
    predicted = comparison_table["Predicted (g/cm³)"]
    weighted_avg = comparison_table["Weighted Avg (g/cm³)"]


    
    plt.figure(figsize=(14, 8))

    # 子图 1: 实际压力
    plt.subplot(1, 2, 1)  # 1行2列，第1个子图
    plt.plot(actual, depth, label='Actual Pressure', linestyle='-', linewidth=0.8, color='blue')
    plt.plot(weighted_avg, depth, label='Transplant Process Pressure', linestyle='-', linewidth=0.8, color='red')
    plt.gca().invert_yaxis()  # 反向 y 轴
    plt.gca().xaxis.tick_top()  # x 轴放在上方
    plt.gca().xaxis.set_label_position('top')  # x 轴标签放在上方
    plt.title('Actual Pressure', fontsize=16, pad=20)
    plt.xlabel('Pressure (g/cm³)', fontsize=14)
    plt.ylabel('Depth (m)', fontsize=14)
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)
    plt.legend(fontsize=12)

    # 子图 2: 预测压力 + 实际压力
    plt.subplot(1, 2, 2)  # 1行2列，第2个子图
    plt.plot(predicted, depth, label='Predicted Pressure', linestyle='-', linewidth=0.8, color='orange')
    plt.plot(actual, depth, label='Actual Pressure', linestyle='-', linewidth=0.8, color='blue')  # 添加实际压力曲线
    plt.gca().invert_yaxis()  # 反向 y 轴
    plt.gca().xaxis.tick_top()  # x 轴放在上方
    plt.gca().xaxis.set_label_position('top')  # x 轴标签放在上方
    plt.title('Predicted Pressure', fontsize=16, pad=20)
    plt.xlabel('Pressure (g/cm³)', fontsize=14)
    plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.7)
    plt.legend(fontsize=12)

    # 保存图片
    output_curve_image_subplots = os.path.join(ewput_file_folder, "ewaa.png")
    plt.savefig(output_curve_image_subplots, bbox_inches='tight', dpi=300)


    # 显示图片
    plt.tight_layout()  # 自动调整子图布局
    plt.show()


    end =time.time() 











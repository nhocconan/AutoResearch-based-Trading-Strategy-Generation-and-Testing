# 6h_Camarilla_R1S1_Breakout_Volume_12hEMA34  
# Hypothesis: 6s/6h uses Camarilla R1/S1 breakout with volume confirmation and 12h EMA34 trend filter.  
# Works in bull/bear via EMA34 trend filter: only long when price > EMA34, short when price < EMA34.  
# Volume > 1.5x 20-period average ensures institutional participation.  
# Target: 50-150 total trades over 4 years (12-37/year)  
# Designed for 6h timeframe to reduce trade frequency vs lower TFs.  

name = "6h_Camarilla_R1S1_Breakout_Volume_12hEMA34"  
timeframe = "6h"  
leverage = 1.0  

import numpy as np  
import pandas as pd  
from mtf_data import get_htf_data, align_htf_to_ltf  

def generate_signals(prices):  
    n = len(prices)  
    if n < 50:  
        return np.zeros(n)  
    
    close = prices['close'].values  
    high = prices['high'].values  
    low = prices['low'].values  
    volume = prices['volume'].values  
    
    # 6h data for EMA34  
    df_6h = get_htf_data(prices, '6h')  
    if len(df_6h) < 34:  
        return np.zeros(n)  
    
    # EMA34 on 6h data  
    ema_6h = pd.Series(df_6h['close']).ewm(span=34, adjust=False).values  
    ema_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_6h)  
    
    # Previous day's Camarilla levels (using 1d data)  
    df_1d = get_htf_data(prices, '1d')  
    if len(df_1d) < 2:  
        return np.zeros(n)  
    
    # Calculate Camarilla levels from previous day  
    ph = df_1d['high'].shift(1).values  # Previous day high  
    pl = df_1d['low'].shift(1).values   # Previous day low  
    pc = df_1d['close'].shift(1).values # Previous day close  
    
    # Camarilla calculations  
    rang = ph - pl  
    r1 = pc + (rang * 1.1 / 12)  
    s1 = pc - (rang * 1.1 / 12)  
    
    # Align Camarilla levels to 6h timeframe  
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)  
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)  
    
    # Volume confirmation: volume > 1.5 * 20-period average  
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  
    volume_confirm = volume > (volume_ma * 1.5)  
    
    signals = np.zeros(n)  
    position = 0  # 0: flat, 1: long, -1: short  
    
    start_idx = max(34, 20)  # Ensure enough data for all indicators  
    
    for i in range(start_idx, n):  
        # Skip if any required data is NaN  
        if (np.isnan(ema_6h_aligned[i]) or np.isnan(r1_aligned[i]) or  
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i])):  
            signals[i] = 0.0  
            continue  
        
        if position == 0:  
            # Long: price breaks above R1 with volume and price > EMA34  
            if (close[i] > r1_aligned[i] and  
                volume_confirm[i] and  
                close[i] > ema_6h_aligned[i]):  
                signals[i] = 0.25  
                position = 1  
            # Short: price breaks below S1 with volume and price < EMA34  
            elif (close[i] < s1_aligned[i] and  
                  volume_confirm[i] and  
                  close[i] < ema_6h_aligned[i]):  
                signals[i] = -0.25  
                position = -1  
                
        elif position == 1:  
            # Long: exit if price breaks below S1  
            if close[i] < s1_aligned[i]:  
                signals[i] = 0.0  
                position = 0  
            else:  
                signals[i] = 0.25  
                
        elif position == -1:  
            # Short: exit if price breaks above R1  
            if close[i] > r1_aligned[i]:  
                signals[i] = 0.0  
                position = 0  
            else:  
                signals[i] = -0.25  
    
    return signals
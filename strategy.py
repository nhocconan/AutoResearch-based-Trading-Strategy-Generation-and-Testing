#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivot levels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Levels (based on previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels using previous day's data
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels (using previous day's values)
    r1 = close_1d + range_hl * 1.1 / 12
    r2 = close_1d + range_hl * 1.1 / 6
    r3 = close_1d + range_hl * 1.1 / 4
    s1 = close_1d - range_hl * 1.1 / 12
    s2 = close_1d - range_hl * 1.1 / 6
    s3 = close_1d - range_hl * 1.1 / 4
    
    # Align to 4h timeframe (use previous day's values - already lagged by calculation)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === Daily ATR for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_ma_30 = pd.Series(atr_10).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_10 / atr_ma_30
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above S1 (support) with volume and moderate volatility
            if (price_close > s1_val and 
                vol_ratio_val > 1.3 and 
                atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R1 (resistance) with volume and moderate volatility
            elif (price_close < r1_val and 
                  vol_ratio_val > 1.3 and 
                  atr_ratio_val > 0.7 and atr_ratio_val < 2.2):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extremes
            if position == 1 and (price_close < s1_val or atr_ratio_val > 2.8 or atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > r1_val or atr_ratio_val > 2.8 or atr_ratio_val < 0.4):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_S1_R1_Breakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_Trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    r1 = close_1d + range_hl * 1.1 / 12
    s1 = close_1d - range_hl * 1.1 / 12
    r2 = close_1d + range_hl * 1.1 / 6
    s2 = close_1d - range_hl * 1.1 / 6
    r3 = close_1d + range_hl * 1.1 / 4
    s3 = close_1d - range_hl * 1.1 / 4
    r4 = close_1d + range_hl * 1.1 / 2
    s4 = close_1d - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 6h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === Daily Trend Filter (EMA34) ===
    close_1d_series = pd.Series(close_1d)
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio with proper initialization
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        ema_34_val = ema_34_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(ema_34_val) or 
            np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(r4_val) or np.isnan(s4_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume and daily uptrend
            if (close_val > r1_val and 
                vol_ratio_val > 1.5 and 
                close_val > ema_34_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and daily downtrend
            elif (close_val < s1_val and 
                  vol_ratio_val > 1.5 and 
                  close_val < ema_34_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Break below S1 or volume dries up
            if close_val < s1_val or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Break above R1 or volume dries up
            if close_val > r1_val or vol_ratio_val < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
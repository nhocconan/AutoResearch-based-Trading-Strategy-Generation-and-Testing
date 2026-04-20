#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivotBreakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Weekly Pivot Points (previous week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for pivot calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Classic pivot (same for standard pivot)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Standard pivot R1 and S1 levels
    r1 = pivot + range_val
    s1 = pivot - range_val
    
    # Align to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # === Volume Trend Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Price Trend Filter: 1d EMA50 > EMA200 for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        ema50_val = ema50[i]
        ema200_val = ema200[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(pivot_val) or 
            np.isnan(ema50_val) or np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume confirmation and uptrend (EMA50 > EMA200)
            if close_val > r1_val and vol_ratio_val > 1.5 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume confirmation and downtrend (EMA50 < EMA200)
            elif close_val < s1_val and vol_ratio_val > 1.5 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR trend breaks down
            if close_val < pivot_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR trend breaks up
            if close_val > pivot_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R2S2_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Close for Trend Filter ===
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_trend_up = weekly_close > weekly_ema
    
    # Align weekly trend to daily
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    
    # === Daily Data (for Pivot Calculation) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla R2 and S2 levels (stronger breakout levels)
    r2 = pivot + (range_val * 1.1 / 6)
    s2 = pivot - (range_val * 1.1 / 6)
    
    # Align to daily timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        pivot_val = pivot_aligned[i]
        weekly_trend = weekly_trend_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r2_val) or 
            np.isnan(s2_val) or np.isnan(pivot_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R2 with volume confirmation and weekly uptrend
            if close_val > r2_val and vol_ratio_val > 1.8 and weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: Break below S2 with volume confirmation and weekly downtrend
            elif close_val < s2_val and vol_ratio_val > 1.8 and not weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR weekly trend turns down
            if close_val < pivot_val or not weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR weekly trend turns up
            if close_val > pivot_val or weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
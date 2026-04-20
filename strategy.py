#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivotBreakout_VolumeTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # === Weekly Pivot Points (previous week) ===
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Previous week's values for pivot calculation
    prev_high = np.roll(high_w, 1)
    prev_low = np.roll(low_w, 1)
    prev_close = np.roll(close_w, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_w[0]
    prev_low[0] = low_w[0]
    prev_close[0] = close_w[0]
    
    # Classic pivot (same for weekly)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Weekly pivot R1 and S1 levels (key levels for daily breakouts)
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    
    # === Daily Volume Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === Daily Price Trend: EMA20 > EMA50 for long, < for short ===
    close_series = pd.Series(prices['close'].values)
    ema20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        pivot_val = pivot_aligned[i]
        ema20_val = ema20[i]
        ema50_val = ema50[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(pivot_val) or 
            np.isnan(ema20_val) or np.isnan(ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly R1 with volume confirmation and uptrend (EMA20 > EMA50)
            if close_val > r1_val and vol_ratio_val > 2.0 and ema20_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S1 with volume confirmation and downtrend (EMA20 < EMA50)
            elif close_val < s1_val and vol_ratio_val > 2.0 and ema20_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below weekly pivot OR trend breaks down
            if close_val < pivot_val or ema20_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above weekly pivot OR trend breaks up
            if close_val > pivot_val or ema20_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
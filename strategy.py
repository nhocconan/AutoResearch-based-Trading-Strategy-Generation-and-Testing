#!/usr/bin/env python3
# 12h_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter
# Hypothesis: Breakouts of weekly Camarilla R1 (resistance) and S1 (support) levels with volume confirmation and 12h EMA trend filter.
# Uses weekly pivots for structure and daily volume for confirmation. Designed for low trade frequency (~15-30/year) to avoid fee drag.
# Trend filter (12h EMA34) ensures alignment with medium-term direction, working in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_R1S1_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate weekly Camarilla R1, S1 levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla R1 = close + (range * 1.1/2), S1 = close - (range * 1.1/2)
    r1_1w = close_1w + (range_1w * 1.1 / 2)
    s1_1w = close_1w - (range_1w * 1.1 / 2)
    
    # === 12h EMA34 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Daily volume ratio (current vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all weekly and 12h levels to 12h
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        r1_1w_val = r1_1w_aligned[i]
        s1_1w_val = s1_1w_aligned[i]
        ema34_12h_val = ema34_12h_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_1w_val) or np.isnan(s1_1w_val) or np.isnan(ema34_12h_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly R1 with volume confirmation, above 12h EMA34
            if (close_val > r1_1w_val and vol_ratio_val > 2.0 and 
                close_val > ema34_12h_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly S1 with volume confirmation, below 12h EMA34
            elif (close_val < s1_1w_val and vol_ratio_val > 2.0 and 
                  close_val < ema34_12h_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below weekly S1
            if close_val <= s1_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to or above weekly R1
            if close_val >= r1_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
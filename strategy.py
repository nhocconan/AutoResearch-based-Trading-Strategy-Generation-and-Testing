#!/usr/bin/env python3
"""
1d_Camarilla_H3_L3_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Daily Camarilla H3/L3 breakout with weekly EMA50 trend filter and volume confirmation on 1d timeframe.
Long when price breaks above H3 (resistance) with volume > 1.3x average and weekly uptrend (close > weekly EMA50).
Short when price breaks below L3 (support) with volume > 1.3x average and weekly downtrend (close < weekly EMA50).
Uses discrete sizing 0.25 to minimize fee churn. Exits on opposite H3/L3 break or trend reversal.
Designed to capture medium-term swings in both bull and bear markets by following the weekly trend while using daily Camarilla levels for precise entries.
Target trades: 15-30/year (60-120 total over 4 years) to stay well below fee drag threshold.
"""

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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla H3 and L3 levels
    H3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 2
    L3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels (based on completed 1d bar)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: 1.3x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly EMA (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        H3_val = H3_aligned[i]
        L3_val = L3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H3 with volume confirmation and weekly uptrend
            long_signal = (high_val > H3_val) and (volume_val > 1.3 * vol_ma_val) and (close_val > ema_50_1w_val)
            # Short: price breaks below L3 with volume confirmation and weekly downtrend
            short_signal = (low_val < L3_val) and (volume_val > 1.3 * vol_ma_val) and (close_val < ema_50_1w_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below L3 or weekly trend turns down
            if low_val < L3_val or close_val < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above H3 or weekly trend turns up
            if high_val > H3_val or close_val > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3_L3_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0
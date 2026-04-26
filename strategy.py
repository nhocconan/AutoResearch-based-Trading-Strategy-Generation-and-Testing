#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirmation
Hypothesis: On daily timeframe, Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume confirmation.
Long when price breaks above R1 with volume > 1.3x average and weekly uptrend (close > weekly EMA50).
Short when price breaks below S1 with volume > 1.3x average and weekly downtrend (close < weekly EMA50).
Uses discrete sizing 0.25 to minimize fee churn. Weekly trend filter reduces whipsaw in bear markets.
Target trades: 15-30/year (60-120 total over 4 years) to stay well below fee drag threshold.
Designed to work in both bull and bear markets by following the weekly trend while using Camarilla levels for precise entries.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous daily OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels from previous daily bar
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels (based on completed daily bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 1.3x average volume (20-day MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly EMA (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
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
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and weekly uptrend
            long_signal = (close_val > R1_val) and (volume_val > 1.3 * vol_ma_val) and (close_val > ema_50_1w_val)
            # Short: price breaks below S1 with volume confirmation and weekly downtrend
            short_signal = (close_val < S1_val) and (volume_val > 1.3 * vol_ma_val) and (close_val < ema_50_1w_val)
            
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
            # Exit: weekly trend reversal or price breaks below S1 (mean reversion)
            if close_val < ema_50_1w_val or close_val < S1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly trend reversal or price breaks above R1 (mean reversion)
            if close_val > ema_50_1w_val or close_val > R1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0
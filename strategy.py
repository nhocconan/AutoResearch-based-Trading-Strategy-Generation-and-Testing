#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrend_HTFRegime_v2
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1w EMA34 trend filter and volume confirmation. Uses weekly trend to capture major moves in both bull/bear markets, with R1/S1 levels for higher-probability breakouts. Volume spike (>1.8x 24-period average) filters noise. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot levels (more stable than lower TF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels using previous 1d bar's OHLC
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_close_1d[0] = df_1d['close'].values[0]
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    range_1d = prev_high_1d - prev_low_1d
    
    # Camarilla levels for 1d - using R1/S1 for breakouts
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)   # R1 level
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)   # S1 level
    
    # Align 1d Camarilla levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume spike confirmation: volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(34) on 1w and volume MA (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend using EMA34
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Breakout logic: trade continuation in direction of 1w trend with volume confirmation
            # In bullish 1w trend: buy R1 breakouts, sell S1 breakdowns
            # In bearish 1w trend: sell S1 breakdowns, buy R1 breakouts
            long_setup = (close[i] > r1_1d_aligned[i]) and htf_1w_bullish and volume_spike[i]
            short_setup = (close[i] < s1_1d_aligned[i]) and htf_1w_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or mean reversion to 1d pivot
            exit_signal = (not htf_1w_bullish) or (close[i] < pivot_1d_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to 1d pivot
            exit_signal = htf_1w_bullish or (close[i] > pivot_1d_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrend_HTFRegime_v2"
timeframe = "12h"
leverage = 1.0
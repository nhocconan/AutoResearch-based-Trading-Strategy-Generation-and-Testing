#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: Trade daily Camarilla R1/S1 breakouts in direction of weekly EMA50 trend with volume confirmation.
Uses 1d primary timeframe for lower trade frequency and 1w trend filter for robustness.
Camarilla levels from prior day provide structure; weekly EMA50 filters for higher timeframe trend alignment; 
volume spike on daily confirms breakout. Works in bull/bear via trend filter + volume confirmation.
Target: 7-25 trades/year per symbol to avoid fee drag.
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Camarilla levels for today (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    camarilla_R1 = close_1d + camarilla_range * 1.0  # R1 = close + 1*(range)
    camarilla_S1 = close_1d - camarilla_range * 1.0  # S1 = close - 1*(range)
    
    # Align Camarilla levels to daily timeframe (prior day's levels available at daily close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 2.0x 20-period average on daily
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), volume MA (20), aligned indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to weekly EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + price above weekly EMA50 + volume spike
            long_breakout = close[i] > camarilla_R1_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i]
            
            # Short: price breaks below Camarilla S1 + price below weekly EMA50 + volume spike
            short_breakout = close[i] < camarilla_S1_aligned[i]
            short_signal = short_breakout and price_below_ema and volume_spike[i]
            
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
            # Exit: price touches Camarilla S1 OR trend turns bearish (price below EMA)
            if (close[i] < camarilla_S1_aligned[i] or not price_above_ema):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R1 OR trend turns bullish (price above EMA)
            if (close[i] > camarilla_R1_aligned[i] or not price_below_ema):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0
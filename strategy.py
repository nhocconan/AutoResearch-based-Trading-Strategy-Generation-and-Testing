#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Trade 12h Camarilla R3/S3 breakouts in direction of 1w EMA50 trend with volume confirmation.
Uses 12h primary timeframe for lower trade frequency (target: 12-37 trades/year) and 1w trend filter for robustness.
Camarilla levels from prior 12h provide structure; 1w EMA50 filters for higher timeframe trend alignment; 
volume spike on 12h confirms breakout. Works in bull/bear via trend filter + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Camarilla levels for today (based on prior 12h OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    camarilla_range = (high_12h - low_12h) * 1.1 / 12.0
    camarilla_R3 = close_12h + camarilla_range * 3.0  # R3 = close + 3*(range)
    camarilla_S3 = close_12h - camarilla_range * 3.0  # S3 = close - 3*(range)
    
    # Align Camarilla levels to 12h timeframe (prior 12h's levels available at 12h close)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S3)
    
    # Volume confirmation: volume > 2.0x 20-period average on 12h
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
            np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: price relative to 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + price above 1w EMA50 + volume spike
            long_breakout = close[i] > camarilla_R3_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i]
            
            # Short: price breaks below Camarilla S3 + price below 1w EMA50 + volume spike
            short_breakout = close[i] < camarilla_S3_aligned[i]
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
            # Exit: price touches Camarilla S3 OR trend turns bearish (price below EMA)
            if (close[i] < camarilla_S3_aligned[i] or not price_above_ema):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches Camarilla R3 OR trend turns bullish (price above EMA)
            if (close[i] > camarilla_R3_aligned[i] or not price_below_ema):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
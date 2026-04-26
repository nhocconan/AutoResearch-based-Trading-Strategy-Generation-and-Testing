#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 6h Camarilla R3/S3 breakouts in direction of 1d EMA34 trend with volume confirmation.
Uses 6h primary timeframe for lower trade frequency and 1d trend filter for robustness.
Camarilla levels from prior 6h provide structure; 1d EMA34 filters for higher timeframe trend alignment; 
volume spike on 6h confirms breakout. Works in bull/bear via trend filter + volume confirmation.
Target: 12-37 trades/year per symbol to avoid fee drag.
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
    
    # Get 6h data for Camarilla calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla levels for today (based on prior 6h OHLC)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    camarilla_range = (high_6h - low_6h) * 1.1 / 12.0
    camarilla_R3 = close_6h + camarilla_range * 3.0  # R3 = close + 3*(range)
    camarilla_S3 = close_6h - camarilla_range * 3.0  # S3 = close - 3*(range)
    
    # Align Camarilla levels to 6h timeframe (prior 6h's levels available at 6h close)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_S3)
    
    # Volume confirmation: volume > 2.0x 20-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34), volume MA (20), aligned indicators
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
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
        
        # Trend filter: price relative to 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + price above 1d EMA34 + volume spike
            long_breakout = close[i] > camarilla_R3_aligned[i]
            long_signal = long_breakout and price_above_ema and volume_spike[i]
            
            # Short: price breaks below Camarilla S3 + price below 1d EMA34 + volume spike
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

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
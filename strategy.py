#!/usr/bin/env python3
"""
6h_VolumeWeighted_SqueezeBreakout
Hypothesis: On 6h timeframe, enter long when price breaks above the upper Bollinger Band with volume > 1.5x 12h average volume and 1d EMA50 uptrend; enter short when price breaks below lower Bollinger Band with volume > 1.5x 12h average volume and 1d EMA50 downtrend. Bollinger Band squeeze (low volatility) precedes breakouts, increasing win rate. Works in bull/bear via 1d trend filter. Target: 20-50 trades/year.
"""

name = "6h_VolumeWeighted_SqueezeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for volume filter (more responsive than 1d)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma20_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h data for Bollinger Bands and price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Bollinger Bands (20), volume MA (20), EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(vol_ma20_12h_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Volume filter: current 6h volume > 1.5x 12h 20-period MA
        volume_filter = volume[i] > vol_ma20_12h_aligned[i] * 1.5
        
        # Trend filter: 1d EMA50 direction
        uptrend_1d = close[i] > ema50_1d_aligned[i]
        downtrend_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: breakout above upper band with volume and uptrend
            if breakout_up and volume_filter and uptrend_1d:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band with volume and downtrend
            elif breakout_down and volume_filter and downtrend_1d:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to middle (SMA20) or breakout fails
            if close[i] <= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle (SMA20) or breakout fails
            if close[i] >= sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
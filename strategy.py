#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Use 1d Camarilla R1/S1 levels for breakout entries, filtered by 1d EMA50 trend direction and volume confirmation.
Camarilla levels provide precise intraday support/resistance; EMA50 filters for trend alignment; volume avoids false breakouts.
Designed for 20-40 trades/year per symbol, works in bull/bear via trend filter.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using prior day)
    high_d = df_1d['high'].shift(1).values
    low_d = df_1d['low'].shift(1).values
    close_d = df_1d['close'].shift(1).values
    
    # Camarilla levels
    range_d = high_d - low_d
    pivot_d = (high_d + low_d + close_d) / 3.0
    r1_d = close_d + range_d * 1.1 / 12
    s1_d = close_d - range_d * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # Daily EMA50 for trend filter
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA50 (50) and volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_d_aligned[i]) or 
            np.isnan(s1_d_aligned[i]) or
            np.isnan(ema50_d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend: price vs EMA50
        bullish_trend = close[i] > ema50_d_aligned[i]
        bearish_trend = close[i] < ema50_d_aligned[i]
        
        if position == 0:
            # Long: bullish daily trend AND price breaks above Camarilla R1 with volume
            if bullish_trend and high[i] > r1_d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish daily trend AND price breaks below Camarilla S1 with volume
            elif bearish_trend and low[i] < s1_d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 OR daily trend turns bearish
            if low[i] < s1_d_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 OR daily trend turns bullish
            if high[i] > r1_d_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
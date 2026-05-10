#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: 4h breakout at daily Camarilla R1/S1 with 12h EMA50 trend filter and volume confirmation.
Uses institutional support/resistance levels and higher timeframe trend to work in both bull/bear.
Target: 20-40 trades/year on 4h to avoid fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get price, volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34) and enough history for Camarilla calculation
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above both EMAs (uptrend) AND price breaks above R1 with volume
            if close[i] > ema_34_1d_aligned[i] and close[i] > ema_50_12h_aligned[i] and high[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below both EMAs (downtrend) AND price breaks below S1 with volume
            elif close[i] < ema_34_1d_aligned[i] and close[i] < ema_50_12h_aligned[i] and low[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR either trend turns bearish
            if low[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR either trend turns bullish
            if high[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
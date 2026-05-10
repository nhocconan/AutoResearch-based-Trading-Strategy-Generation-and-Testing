#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_1wVolume
Hypothesis: On 12h timeframe, use Camarilla R1/S1 breakout from previous 1d for entry, filtered by 1d EMA34 trend and 1w volume spike. This combines price structure (Camarilla pivots) with trend and volume confirmation to capture multi-day moves. Target 15-30 trades/year to minimize fee drag. Works in bull/bear via trend filter and volume confirmation.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_1wVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    R1 = close + (range_val * 1.1 / 12)
    S1 = close - (range_val * 1.1 / 12)
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's HLC)
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_S1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        R1, S1 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        camarilla_R1[i] = R1
        camarilla_S1[i] = S1
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h, 2 bars per day)
    camarilla_R1_12h = align_ltf_to_htf(prices, df_1d, camarilla_R1)
    camarilla_S1_12h = align_ltf_to_htf(prices, df_1d, camarilla_S1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_12h = align_ltf_to_htf(prices, df_1d, ema34_1d)
    
    # Get 1w data for volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1w_12h = align_ltf_to_htf(prices, df_1w, vol_ma20_1w)
    
    # 12h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d), EMA34 (1d), volume MA (1w)
    start_idx = 34  # EMA34 needs 34 bars
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_12h[i]) or 
            np.isnan(camarilla_S1_12h[i]) or
            np.isnan(ema34_1d_12h[i]) or
            np.isnan(vol_ma20_1w_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs 1d EMA34
        uptrend_1d = close[i] > ema34_1d_12h[i]
        downtrend_1d = close[i] < ema34_1d_12h[i]
        
        # Volume filter: current 12h volume > 1.5x 1w 20-period MA
        volume_filter = volume[i] > vol_ma20_1w_12h[i] * 1.5
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend with volume
            if close[i] > camarilla_R1_12h[i] and uptrend_1d and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 in downtrend with volume
            elif close[i] < camarilla_S1_12h[i] and downtrend_1d and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 or trend fails
            if close[i] < camarilla_S1_12h[i] or not uptrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 or trend fails
            if close[i] > camarilla_R1_12h[i] or not downtrend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
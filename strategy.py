#!/usr/bin/env python3
# 4h_1d_camarilla_pivot_volume_v1
# Strategy: 4h Camarilla pivot level touch with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels from daily chart act as strong support/resistance.
# Price touching these levels with volume confirmation and daily trend alignment
# provides high-probability mean-reversion entries. Works in both bull and bear
# markets by fading extremes in trending environments. Designed for low trade
# frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Formula: Range = high - low
    # H4 = close + 1.5 * (high - low) / 2
    # L4 = close - 1.5 * (high - low) / 2
    # H3 = close + 1.125 * (high - low) / 2
    # L3 = close - 1.125 * (high - low) / 2
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * range_1d / 2
    camarilla_l4 = close_1d - 1.5 * range_1d / 2
    camarilla_h3 = close_1d + 1.125 * range_1d / 2
    camarilla_l3 = close_1d - 1.125 * range_1d / 2
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar to close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Price proximity to Camarilla levels (within 0.1% tolerance)
        tol = 0.001  # 0.1% tolerance
        near_h4 = abs(high[i] - h4_aligned[i]) / h4_aligned[i] < tol
        near_l4 = abs(low[i] - l4_aligned[i]) / l4_aligned[i] < tol
        near_h3 = abs(high[i] - h3_aligned[i]) / h3_aligned[i] < tol
        near_l3 = abs(low[i] - l3_aligned[i]) / l3_aligned[i] < tol
        
        # 1d EMA trend filter
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price touches L3/L4 AND bullish daily trend AND volume confirmation
        if ((near_l3 or near_l4) and trend_bullish and vol_confirm and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Price touches H3/H4 AND bearish daily trend AND volume confirmation
        elif ((near_h3 or near_h4) and trend_bearish and vol_confirm and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Price touches opposite level or trend changes
        elif position == 1 and (near_h3 or near_h4):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (near_l3 or near_l4):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
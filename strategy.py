#!/usr/bin/env python3
"""
6h_Structure_Volume_Reversal
Hypothesis: In 6-hour timeframe, mean-reversion opportunities occur when price deviates
significantly from structure (recent swing high/low) with volume confirmation.
Longs: price near swing low + volume spike + bullish pressure.
Shorts: price near swing high + volume spike + bearish pressure.
Works in both bull and bear markets by capturing exhaustion moves.
Uses 1-day swing structure for robustness and 1-week trend filter to avoid counter-trend trades.
Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for swing structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate swing high/low on 1d (10-period lookback)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 10-period rolling max/min for swing levels
    swing_high_1d = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    swing_low_1d = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # Align swing levels to 6h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high_1d)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low_1d)
    
    # Distance from swing levels as percentage of range
    swing_range = swing_high_aligned - swing_low_aligned
    # Avoid division by zero
    swing_range = np.where(swing_range == 0, 1e-10, swing_range)
    dist_from_low = (close - swing_low_aligned) / swing_range  # 0 at low, 1 at high
    dist_from_high = (swing_high_aligned - close) / swing_range  # 0 at high, 1 at low
    
    # Volume filter: current volume > 1.8x 20-period average (signifies exhaustion)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1-week EMA trend filter (21-period)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_1w_aligned[i]
        vol_spike = volume_spike[i]
        dist_low = dist_from_low[i]
        dist_high = dist_from_high[i]
        
        # Entry conditions: near swing extremes with volume spike
        near_low = dist_low < 0.15  # Within 15% of swing low
        near_high = dist_high < 0.15  # Within 15% of swing high
        
        if position == 0:
            # Long: near swing low + volume spike + above weekly trend
            if near_low and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: near swing high + volume spike + below weekly trend
            elif near_high and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: return to midpoint or trend breaks
            midpoint = (swing_low_aligned[i] + swing_high_aligned[i]) / 2
            if price > midpoint or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: return to midpoint or trend breaks
            midpoint = (swing_low_aligned[i] + swing_high_aligned[i]) / 2
            if price < midpoint or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Structure_Volume_Reversal"
timeframe = "6h"
leverage = 1.0
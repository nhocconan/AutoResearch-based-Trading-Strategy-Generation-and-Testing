#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: Donchian(20) breakouts on 12h with 1d EMA50 trend filter and volume confirmation. Works in bull (breakouts continue uptrend) and bear (breakouts continue downtrend) by following the 1d trend. Targets 12-37 trades/year.
"""

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d EMA50 trend filter ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Donchian(20) channels on 12h ---
    # Highest high of last 20 periods
    highest_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    lowest_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # --- 12h volume confirmation ---
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Hold position until exit signal
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 12h average
        vol_confirm = volume_12h[i] > 1.5 * vol_avg_12h[i]
        
        if position == 0:
            # Look for breakout entries in direction of 1d trend
            if close_12h[i] > highest_20[i] and close_12h[i] > ema50_1d_aligned[i] and vol_confirm:
                # Bullish breakout above Donchian high with uptrend and volume
                signals[i] = 0.25
                position = 1
            elif close_12h[i] < lowest_20[i] and close_12h[i] < ema50_1d_aligned[i] and vol_confirm:
                # Bearish breakdown below Donchian low with downtrend and volume
                signals[i] = -0.25
                position = -1
        else:
            # Manage existing position: exit on opposite Donchian breach
            if position == 1:
                # Long: exit when price breaks below Donchian low
                if close_12h[i] < lowest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit when price breaks above Donchian high
                if close_12h[i] > highest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_Volume
Hypothesis: 12h Donchian(20) breakouts in the direction of the 1d EMA50 trend with volume confirmation capture momentum in both bull and bear markets. Uses 1d trend filter to avoid counter-trend trades, reducing whipsaws in sideways/ bear markets. Volume confirmation ensures breakouts have participation. Targets 50-150 total trades over 4 years.
"""

name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # --- 1d EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 12h Donchian(20) channels ---
    # Upper channel: highest high of last 20 periods
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # --- 12h Volume confirmation ---
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR from entry
                atr_est = np.abs(high_12h[i] - low_12h[i])  # rough 12h ATR estimate
                if position == 1 and close_12h[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.3x 12h average
        vol_confirm = volume_12h[i] > 1.3 * vol_avg_12h[i]
        
        if position == 0:
            # Look for breakout entries in direction of 1d trend
            if vol_confirm:
                # Long breakout: price above upper Donchian and above 1d EMA50
                if close_12h[i] > highest_high[i] and close_12h[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_12h[i]
                # Short breakdown: price below lower Donchian and below 1d EMA50
                elif close_12h[i] < lowest_low[i] and close_12h[i] < ema50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit on breakdown below lower Donchian
                if close_12h[i] < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on breakout above upper Donchian
                if close_12h[i] > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
12h_TripleConfirmation_Breakout
Hypothesis: Combines 12h Donchian(20) breakout with 1d volume spike and 1w trend filter.
This creates a high-conviction signal that works in both bull and bear markets by requiring:
1) Price breakout from 20-period channel (momentum)
2) Volume > 2x 20-period average (conviction)
3) Price above/below 200-period 1w EMA (trend alignment)
Limits trades to ~15-25/year to avoid fee drag while capturing strong moves.
"""

name = "12h_TripleConfirmation_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Volume Spike (20-period average) ---
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / (vol_avg_1d + 1e-10)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # --- 1w EMA200 for trend filter ---
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # --- 12h Donchian Channel (20-period) ---
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 200  # for EMA200 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
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
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        vol_spike = vol_ratio_12h_aligned[i] > 2.0
        
        if position == 0:
            # Look for breakout entries with volume spike and trend alignment
            if vol_spike:
                # Long breakout: price above Donchian high AND above 1w EMA200
                if close_12h[i] > highest_high[i] and close_12h[i] > ema200_1w_aligned[i]:
                    signals[i] = 0.25  # long breakout
                    position = 1
                    entry_price = close_12h[i]
                # Short breakdown: price below Donchian low AND below 1w EMA200
                elif close_12h[i] < lowest_low[i] and close_12h[i] < ema200_1w_aligned[i]:
                    signals[i] = -0.25  # short breakdown
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position: exit on opposite Donchian touch
            if position == 1:
                # Long: exit when price touches or crosses Donchian low
                if close_12h[i] <= lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit when price touches or crosses Donchian high
                if close_12h[i] >= highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
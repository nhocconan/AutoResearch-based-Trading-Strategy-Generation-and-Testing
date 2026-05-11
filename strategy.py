#!/usr/bin/env python3
"""
6H_WeeklyPivot_DonchianBreakout_Volume
Hypothesis: Price breaking above/below weekly pivot-derived support/resistance (R1/S1) with 
Donchian(20) breakout in same direction and volume confirmation, filtered by daily trend (price > EMA50).
Weekly pivots capture institutional levels; Donchian breakouts signal momentum; volume confirms participation.
Daily trend filter avoids counter-trend whipsaws. Designed for low frequency (15-30 trades/year) 
to work in both bull (breakouts) and bear (mean reversion at extremes) markets.
"""

name = "6H_WeeklyPivot_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation (using weekly high/low/close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly Pivot Points (R1, S1) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pp_w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1 and S1
    r1_w = 2 * pp_w - low_1w
    s1_w = 2 * pp_w - high_1w
    
    # Align weekly levels to 6h timeframe (using previous week's levels)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # --- Daily EMA50 for trend filter ---
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Donchian Channel (20) on 6h ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- Volume Spike (6h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: Price > weekly R1 AND breaks Donchian high AND volume spike AND above daily EMA50
        # Short: Price < weekly S1 AND breaks Donchian low AND volume spike AND below daily EMA50
        long_entry = (close[i] > r1_w_aligned[i]) and \
                     (high[i] > highest_high[i-1]) and \
                     vol_spike[i] and \
                     (close[i] > ema_50_1d_aligned[i])
        
        short_entry = (close[i] < s1_w_aligned[i]) and \
                      (low[i] < lowest_low[i-1]) and \
                      vol_spike[i] and \
                      (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: 
            # Long: Price crosses below weekly pivot OR Donchian low OR below daily EMA50
            # Short: Price crosses above weekly pivot OR Donchian high OR above daily EMA50
            if position == 1:
                pp_w_aligned = align_htf_to_ltf(prices, df_1w, pp_w)
                if (close[i] < pp_w_aligned[i]) or \
                   (low[i] < lowest_low[i]) or \
                   (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                pp_w_aligned = align_htf_to_ltf(prices, df_1w, pp_w)
                if (close[i] > pp_w_aligned[i]) or \
                   (high[i] > highest_high[i]) or \
                   (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
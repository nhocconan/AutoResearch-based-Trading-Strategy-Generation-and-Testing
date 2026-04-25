#!/usr/bin/env python3
"""
6h Williams Fractal Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Williams fractals identify swing highs/lows. Breakout above recent bullish fractal or below bearish fractal with volume confirmation and weekly pivot trend filter captures momentum. Works in bull (long on bullish fractal break) and bear (short on bearish fractal break). Weekly pivot ensures we trade with the higher timeframe trend. Target: 12-37 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    # Weekly trend: price above pivot = bullish, below = bearish
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Get 1d data for Williams fractals (more reliable than lower TF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 20-period volume MA for volume confirmation (on 6h data)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for volume MA and fractals
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        weekly_pivot = weekly_pivot_aligned[i]
        bullish_fractal = bullish_fractal_aligned[i]
        bearish_fractal = bearish_fractal_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price > bullish fractal, above weekly pivot, volume confirmation
            long_entry = (curr_close > bullish_fractal) and (curr_close > weekly_pivot) and volume_confirm
            # Short: price < bearish fractal, below weekly pivot, volume confirmation
            short_entry = (curr_close < bearish_fractal) and (curr_close < weekly_pivot) and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below weekly pivot OR price breaks below bearish fractal (stop and reverse)
            if curr_close < weekly_pivot or curr_close < bearish_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above weekly pivot OR price breaks above bullish fractal (stop and reverse)
            if curr_close > weekly_pivot or curr_close > bullish_fractal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_Fractal_Breakout_WeeklyPivot_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
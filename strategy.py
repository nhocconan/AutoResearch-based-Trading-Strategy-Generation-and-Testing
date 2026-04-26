#!/usr/bin/env python3
"""
12h_WilliamsFractal_Breakout_1wTrend_VolumeFilter
Hypothesis: Williams Fractal breakouts (bullish/bearish) on 12h timeframe with 1w EMA50 trend filter and volume spike confirmation.
Only go long when bullish fractal breakout occurs AND price > 1w EMA50 AND volume spike.
Only go short when bearish fractal breakout occurs AND price < 1w EMA50 AND volume spike.
Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year per symbol.
Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
Williams Fractals require 2-bar confirmation delay on HTF to avoid look-ahead.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Williams Fractals (requires 2-bar confirmation delay)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Compute Williams Fractals on 1d timeframe
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Williams Fractals need 2 extra 1d bars for confirmation (already handled in compute_williams_fractals)
    # Apply additional 2-bar delay for confirmation as per Rule 2b
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume spike detector (30-bar volume MA on 12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Bullish fractal breakout, price above 1w EMA50, volume spike
            if bullish_fractal_aligned[i] and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal breakout, price below 1w EMA50, volume spike
            elif bearish_fractal_aligned[i] and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bearish fractal occurs OR trend changes to downtrend
            if bearish_fractal_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bullish fractal occurs OR trend changes to uptrend
            if bullish_fractal_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WilliamsFractal_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0
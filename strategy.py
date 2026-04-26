#!/usr/bin/env python3
"""
6h_WilliamsFractal_WeeklyTrend_VolumeFilter
Hypothesis: Williams fractals on 1d identify swing points; weekly trend filter ensures trades align with higher timeframe momentum; volume confirmation reduces false breakouts. Works in bull/bear by trading breakouts in direction of weekly trend with volume validation.
Timeframe: 6h, HTF: 1w (trend) and 1d (fractals)
Target: 12-30 trades/year per symbol (50-120 over 4 years)
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams fractals on daily (requires 2-bar confirmation after center)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Align with 2-bar extra delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Volume spike detector (50-period volume MA on 6h)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA + 2 for fractal alignment)
    start_idx = 52
    
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
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Bullish fractal break (price above recent high) with volume spike and weekly uptrend
            if close[i] > bullish_fractal_aligned[i] and volume_spike[i] and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish fractal break (price below recent low) with volume spike and weekly downtrend
            elif close[i] < bearish_fractal_aligned[i] and volume_spike[i] and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Weekly trend turns down OR price re-enters below bullish fractal level
            if not weekly_uptrend or close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Weekly trend turns up OR price re-enters above bearish fractal level
            if not weekly_downtrend or close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0
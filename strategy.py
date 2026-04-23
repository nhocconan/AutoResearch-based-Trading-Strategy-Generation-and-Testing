#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Fractal breakout with weekly EMA50 trend filter and volume confirmation
- Long when bullish fractal breaks above weekly EMA50 AND volume > 1.5x 20-period average
- Short when bearish fractal breaks below weekly EMA50 AND volume > 1.5x 20-period average
- Exit when price crosses the weekly EMA50 (mean reversion to trend)
- Uses 1w EMA50 for HTF trend alignment to avoid counter-trend entries
- Volume confirmation ensures institutional participation and reduces false breakouts
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Williams fractal needs 2 extra 1w bars after the center bar for confirmation
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1w['high'].values,
        df_1w['low'].values,
    )
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1w, bullish_fractal, additional_delay_bars=2
    )
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 51)  # Need 20 for volume MA, 51 for EMA50 (50+1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using Williams fractal levels)
        bullish_breakout = bullish_fractal_aligned[i] > ema50_1w_aligned[i]  # Bullish fractal above EMA50
        bearish_breakout = bearish_fractal_aligned[i] < ema50_1w_aligned[i]  # Bearish fractal below EMA50
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish fractal breakout + volume confirmation
            if bullish_breakout and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal breakout + volume confirmation
            elif bearish_breakout and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses the weekly EMA50 (mean reversion to trend)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below weekly EMA50
                if close[i] < ema50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above weekly EMA50
                if close[i] > ema50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0
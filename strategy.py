#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsFractal_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA20 trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w = (close_1w > ema20_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Get daily data once for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Williams Fractals calculation
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_vals = df_1d['high'].values
    low_vals = df_1d['low'].values
    n_1d = len(high_vals)
    
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        if (high_vals[i-2] < high_vals[i-1] and 
            high_vals[i] < high_vals[i-1] and
            high_vals[i-3] < high_vals[i-1] and
            high_vals[i+1] < high_vals[i-1]):
            bearish_fractal[i] = True
            
        if (low_vals[i-2] > low_vals[i-1] and 
            low_vals[i] > low_vals[i-1] and
            low_vals[i-3] > low_vals[i-1] and
            low_vals[i+1] > low_vals[i-1]):
            bullish_fractal[i] = True
    
    # Align fractals to 6h with additional delay for confirmation
    # Williams fractal needs 2 extra daily bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2
    )
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: bullish fractal confirmed + weekly uptrend + volume spike
            long_cond = (bullish_fractal_aligned[i] > 0.5 and 
                        trend_1w_aligned[i] > 0.5 and 
                        vol_spike[i])
            
            # Short entry: bearish fractal confirmed + weekly downtrend + volume spike
            short_cond = (bearish_fractal_aligned[i] > 0.5 and 
                         trend_1w_aligned[i] < 0.5 and 
                         vol_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish fractal confirmed (reversal signal)
            if bearish_fractal_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish fractal confirmed (reversal signal)
            if bullish_fractal_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Fractal reversal signals on 6h timeframe with weekly EMA20 trend filter and volume spike confirmation.
# Williams Fractals identify potential reversal points - bullish fractals suggest bottoms, bearish fractals suggest tops.
# Weekly trend filter ensures we trade in the direction of the higher timeframe trend.
# Volume spike confirmation reduces false signals by requiring increased participation at fractal points.
# Designed to work in both bull (buy bullish fractals in uptrend) and bear (sell bearish fractals in downtrend) markets.
# Discrete sizing (0.25) minimizes churn. Target: 15-35 trades/year.
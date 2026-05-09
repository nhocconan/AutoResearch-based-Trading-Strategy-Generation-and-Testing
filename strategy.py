#!/usr/bin/env python3
# Hypothesis: 4h Williams Fractal breakout with 1d EMA50 trend filter and volume spike
# Long when price breaks above bullish fractal with EMA50 uptrend and volume > 2x average
# Short when price breaks below bearish fractal with EMA50 downtrend and volume > 2x average
# Exit when price crosses opposite fractal or reverses to 50% retracement level
# Williams Fractals identify key swing points with built-in confirmation (requires 2 bars after)
# Designed to capture breakouts at swing highs/lows with institutional significance
# Target: 80-140 total trades over 4 years (20-35/year) with size 0.25

name = "4h_WilliamsFractal_Breakout_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals (requires 5-bar window)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 after)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above bullish fractal, EMA50 uptrend, volume spike
            if (close[i] > bullish_fractal_aligned[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below bearish fractal, EMA50 downtrend, volume spike
            elif (close[i] < bearish_fractal_aligned[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below bearish fractal or retraces 50% to bullish fractal
            if (close[i] < bearish_fractal_aligned[i] or 
                close[i] < (bullish_fractal_aligned[i] + bearish_fractal_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above bullish fractal or retraces 50% to bearish fractal
            if (close[i] > bullish_fractal_aligned[i] or 
                close[i] > (bullish_fractal_aligned[i] + bearish_fractal_aligned[i]) / 2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
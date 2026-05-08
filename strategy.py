#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Fractal breakout with weekly trend filter and volume confirmation
# We go long when price breaks above a bearish fractal resistance level with weekly EMA(34) uptrend and volume spike.
# We go short when price breaks below a bullish fractal support level with weekly EMA(34) downtrend and volume spike.
# Uses 6h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Williams Fractals provide natural support/resistance levels that work in both trending and ranging markets.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the breakout.

name = "6h_WilliamsFractal_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractals
        return np.zeros(n)
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-2] and high[n+1] < high[n]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-2] and low[n+1] > low[n]
    high_vals = df_1d['high'].values
    low_vals = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_vals), np.nan)
    bullish_fractal = np.full(len(low_vals), np.nan)
    
    # Need at least 2 points on each side
    for i in range(2, len(high_vals) - 2):
        # Bearish fractal: highest high in the middle
        if (high_vals[i] > high_vals[i-1] and high_vals[i] > high_vals[i-2] and
            high_vals[i] > high_vals[i+1] and high_vals[i] > high_vals[i+2]):
            bearish_fractal[i] = high_vals[i]
        
        # Bullish fractal: lowest low in the middle
        if (low_vals[i] < low_vals[i-1] and low_vals[i] < low_vals[i-2] and
            low_vals[i] < low_vals[i+1] and low_vals[i] < low_vals[i+2]):
            bullish_fractal[i] = low_vals[i]
    
    # Apply additional delay of 2 days for fractal confirmation (needs 2 future candles to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume spike: current volume > 2.0 * 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        bearish_fractal_val = bearish_fractal_aligned[i]
        bullish_fractal_val = bullish_fractal_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above bearish fractal resistance + weekly uptrend + volume spike
            if (not np.isnan(bearish_fractal_val) and close[i] > bearish_fractal_val and 
                close[i] > ema34_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below bullish fractal support + weekly downtrend + volume spike
            elif (not np.isnan(bullish_fractal_val) and close[i] < bullish_fractal_val and 
                  close[i] < ema34_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish fractal support OR weekly trend turns down
            if (not np.isnan(bullish_fractal_val) and close[i] < bullish_fractal_val) or close[i] < ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish fractal resistance OR weekly trend turns up
            if (not np.isnan(bearish_fractal_val) and close[i] > bearish_fractal_val) or close[i] > ema34_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
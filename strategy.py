#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal reversal with 1d trend filter and volume confirmation
# Long when bullish fractal forms at support, 1d EMA34 rising, volume > 1.3x average
# Short when bearish fractal forms at resistance, 1d EMA34 falling, volume > 1.3x average
# Uses 6h for entry timing, 1d for trend filter to avoid whipsaws in choppy markets
# Targets 50-150 total trades over 4 years (12-37/year) for low fee drag and high win rate
# Williams fractal requires 2-bar confirmation after the center bar

name = "6h_WilliamsFractal_1dTrend_Volume"
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
    
    # Get 1d data for Williams fractal and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need at least 10 days for fractal calculation
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams fractal: 5-point pattern (high/low surrounded by 2 lower/higher on each side)
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high[i] is highest among i-2, i-1, i, i+1, i+2
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: low[i] is lowest among i-2, i-1, i, i+1, i+2
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Williams fractal requires 2-bar confirmation after the center bar
    # So we shift the signal by 2 bars to the right (future confirmation)
    bearish_fractal_confirmed = np.zeros_like(bearish_fractal)
    bullish_fractal_confirmed = np.zeros_like(bullish_fractal)
    bearish_fractal_confirmed[2:] = bearish_fractal[:-2]
    bullish_fractal_confirmed[2:] = bullish_fractal[:-2]
    
    # Get 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed.astype(float))
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed.astype(float))
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bearish_fractal_val = bearish_fractal_aligned[i] > 0.5
        bullish_fractal_val = bullish_fractal_aligned[i] > 0.5
        ema34_1d_val = ema34_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: bullish fractal forms, 1d uptrend, volume spike
            if bullish_fractal_val and ema34_1d_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish fractal forms, 1d downtrend, volume spike
            elif bearish_fractal_val and ema34_1d_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish fractal forms or 1d trend down
            if bearish_fractal_val or ema34_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish fractal forms or 1d trend up
            if bullish_fractal_val or ema34_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
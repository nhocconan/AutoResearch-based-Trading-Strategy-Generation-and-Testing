#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 1w EMA50 trend filter and volume confirmation
# Uses Williams Fractals to identify swing points and breakouts in direction of weekly trend
# Weekly EMA50 ensures alignment with major trend to reduce counter-trend signals
# Volume confirmation at 1.8x average filters low-participation moves
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag while allowing multiple entries per year
# Works in both bull and bear markets by combining trend filter with momentum oscillators

name = "6h_WilliamsFractal_Breakout_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Fractals (5-bar: 2 left, 2 right)
    # Bullish fractal: low[i] is lowest of 5 bars (i-2, i-1, i, i+1, i+2)
    # Bearish fractal: high[i] is highest of 5 bars (i-2, i-1, i, i+1, i+2)
    bullish_fractal = np.zeros(n, dtype=bool)
    bearish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        # Bullish fractal: current low is lowest in window
        if (low[i] <= low[i-2] and low[i] <= low[i-1] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish_fractal[i] = True
        
        # Bearish fractal: current high is highest in window
        if (high[i] >= high[i-2] and high[i] >= high[i-1] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish_fractal[i] = True
    
    # Volume confirmation: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish fractal breakout AND price > 1w EMA50 (uptrend) AND volume spike
            if (bullish_fractal[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish fractal breakout AND price < 1w EMA50 (downtrend) AND volume spike
            elif (bearish_fractal[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish fractal formed (potential top) OR price < 1w EMA50 (trend change)
            if bearish_fractal[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish fractal formed (potential bottom) OR price > 1w EMA50 (trend change)
            if bullish_fractal[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
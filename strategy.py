#!/usr/bin/env python3
name = "12h_WilliamsFractal_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA100 for trend filter
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    
    # Load daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Fractals (requires 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_100_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish fractal + above weekly EMA100 + volume spike
            if bullish_fractal_aligned[i] == 1.0 and close[i] > ema_100_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish fractal + below weekly EMA100 + volume spike
            elif bearish_fractal_aligned[i] == 1.0 and close[i] < ema_100_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish fractal or below weekly EMA100
            if bearish_fractal_aligned[i] == 1.0 or close[i] < ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish fractal or above weekly EMA100
            if bullish_fractal_aligned[i] == 1.0 or close[i] > ema_100_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
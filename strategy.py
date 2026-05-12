#!/usr/bin/env python3
name = "6h_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data for trend filter and fractal calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Fractals on daily data (need 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Bearish fractal: potential resistance (look for breakdowns)
    # Bullish fractal: potential support (look for breakouts)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume filter: current volume > 1.8x 30-period average (5 days of 6h data)
    vol_avg = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above bullish fractal (support) + above daily EMA34 + volume filter
            if high[i] > bullish_fractal_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below bearish fractal (resistance) + below daily EMA34 + volume filter
            elif low[i] < bearish_fractal_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakdown below bullish fractal or below daily EMA34
            if low[i] < bullish_fractal_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout above bearish fractal or above daily EMA34
            if high[i] > bearish_fractal_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
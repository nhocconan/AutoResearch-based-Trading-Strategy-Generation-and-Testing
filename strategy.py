#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Williams Fractal for market structure and 1-week EMA for trend direction.
# Williams Fractals identify swing highs/lows with confirmation delay. Price breaking above a bearish fractal
# suggests bullish continuation; breaking below a bullish fractal suggests bearish continuation.
# 1-week EMA filter ensures trades align with higher timeframe trend, reducing counter-trend whipsaw.
# Volume confirmation (>1.5x 20-period average) filters low-probability breakouts.
# Designed to work in both bull and bear markets by using weekly trend filter.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high[i] is highest among 5 bars (i-2, i-1, i, i+1, i+2)
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal: low[i] is lowest among 5 bars
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate EMA(21) on 1w close
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align indicators to 6h timeframe
    # Williams Fractals need 2-bar confirmation delay (already calculated, but align with extra delay for safety)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # EMA alignment (no extra delay needed as it's based on weekly close)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 5)  # Need volume MA and fractal lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts with trend filter
            # Long: price above bearish fractal AND price > weekly EMA (uptrend)
            if (not np.isnan(bearish_fractal_aligned[i]) and
                close[i] > bearish_fractal_aligned[i] and
                close[i] > ema_21_1w_aligned[i] and
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below bullish fractal AND price < weekly EMA (downtrend)
            elif (not np.isnan(bullish_fractal_aligned[i]) and
                  close[i] < bullish_fractal_aligned[i] and
                  close[i] < ema_21_1w_aligned[i] and
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to bullish fractal or weekly EMA turns down
            if (not np.isnan(bullish_fractal_aligned[i]) and
                close[i] < bullish_fractal_aligned[i]) or \
               (i > 0 and ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to bearish fractal or weekly EMA turns up
            if (not np.isnan(bearish_fractal_aligned[i]) and
                close[i] > bearish_fractal_aligned[i]) or \
               (i > 0 and ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dWilliamsFractal_1wEMA_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0
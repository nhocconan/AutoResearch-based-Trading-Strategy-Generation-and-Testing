#!/usr/bin/env python3

# 6h_WilliamsFractal_12hTrend_VolumeFilter
# Hypothesis: Williams fractal reversals from daily timeframe, filtered by 12h trend and volume, 
# capture high-probability reversals in both bull and bear markets. Fractals require 
# two-bar confirmation (no look-ahead), 12h EMA50 filters trend direction, and volume spike 
# confirms momentum. Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

name = "6h_WilliamsFractal_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams fractals: bearish (high) and bullish (low)"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest of 5 bars (i-2 to i+2)
        if (high[i] >= high[i-1] and high[i] >= high[i-2] and 
            high[i] >= high[i+1] and high[i] >= high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest of 5 bars (i-2 to i+2)
        if (low[i] <= low[i-1] and low[i] <= low[i-2] and 
            low[i] <= low[i+1] and low[i] <= low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Williams fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)

    # Calculate Williams fractals on daily data
    bearish_fractal, bullish_fractal = calculate_williams_fractals(
        df_1d['high'].values, df_1d['low'].values
    )
    
    # Align fractals to 6h timeframe with 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    # Calculate 12h EMA50 for trend filter
    ema_12h = pd.Series(df_12h['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish fractal with 12h uptrend and volume spike
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] > bullish_fractal_aligned[i] and
                close[i] > ema_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish fractal with 12h downtrend and volume spike
            elif (not np.isnan(bearish_fractal_aligned[i]) and 
                  close[i] < bearish_fractal_aligned[i] and
                  close[i] < ema_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below bullish fractal or 12h trend turns down
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] < bullish_fractal_aligned[i]) or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above bearish fractal or 12h trend turns up
            if (not np.isnan(bearish_fractal_aligned[i]) and 
                close[i] > bearish_fractal_aligned[i]) or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
#!/usr/bin/env python3
# 4h_1d_WilliamsFractal_Breakout_TrendFilter
# Hypothesis: Uses daily Williams Fractal breakouts with 1d trend filter to capture multi-day momentum.
# Enters long when price breaks above the most recent bearish fractal (resistance) with 1d uptrend.
# Enters short when price breaks below the most recent bullish fractal (support) with 1d downtrend.
# Williams Fractals require 2-bar confirmation, so we use additional_delay_bars=2 when aligning.
# Designed for low trade frequency (~20-60 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following 1d trend while using fractal breakouts for precise entries.

name = "4h_1d_WilliamsFractal_Breakout_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Daily data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (require 2-bar confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        high_1d, low_1d
    )
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align fractals to 4h timeframe with 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above bearish fractal (resistance) + 1d EMA50 uptrend
            if (close[i] > bearish_fractal_aligned[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bullish fractal (support) + 1d EMA50 downtrend
            elif (close[i] < bullish_fractal_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below bullish fractal (support) OR closes below 1d EMA50
            if (close[i] < bullish_fractal_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above bearish fractal (resistance) OR closes above 1d EMA50
            if (close[i] > bearish_fractal_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
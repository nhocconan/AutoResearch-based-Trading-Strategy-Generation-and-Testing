#!/usr/bin/env python3
"""
12h_WilliamsFractal_Breakout_1dTrend_Volume
Hypothesis: Daily Williams Fractals identify key support/resistance levels.
Breakouts above bearish fractals or below bullish fractals with volume confirmation
and daily trend alignment capture momentum moves. Exit on trend reversal or
fractal roll. Designed for 12h timeframe to limit trades (~15-25/year) and
perform in both bull and bear markets via trend filter and volatility-adjusted stops.
"""

name = "12h_WilliamsFractal_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    # Williams Fractals require 2 extra daily bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (12 days on 12h)
    vol_ma = pd.Series(volume).rolling(
        window=24, min_periods=24
    ).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above bearish fractal with volume and uptrend
            if (bearish_fractal_aligned[i] > 0 and  # valid fractal
                close[i] > bearish_fractal_aligned[i] and
                volume_filter[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below bullish fractal with volume and downtrend
            elif (bullish_fractal_aligned[i] > 0 and  # valid fractal
                  close[i] < bullish_fractal_aligned[i] and
                  volume_filter[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price below bullish fractal (support)
            if (close[i] < ema34_1d_aligned[i]) or \
               (bullish_fractal_aligned[i] > 0 and close[i] < bullish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price above bearish fractal (resistance)
            if (close[i] > ema34_1d_aligned[i]) or \
               (bearish_fractal_aligned[i] > 0 and close[i] > bearish_fractal_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
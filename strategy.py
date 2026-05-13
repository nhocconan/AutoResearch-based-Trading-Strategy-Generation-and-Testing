#!/usr/bin/env python3
# 6h_WilliamsFractal_1dTrend_VolumeBreakout
# Hypothesis: Enter long when Williams bearish fractal breaks (price > bearish fractal level) in the direction of 1d EMA50 trend with volume spike.
# Enter short when bullish fractal breaks (price < bullish fractal level) in the direction of 1d EMA50 trend with volume spike.
# Williams fractals identify potential reversal points. A break of the fractal level indicates continuation with momentum.
# Volume surge confirms institutional participation. Trend filter ensures alignment with higher timeframe momentum.
# Works in bull (breakouts above fractal in uptrend) and bear (breakdowns below fractal in downtrend).
# Low frequency due to fractal break requirement and strict volume confirmation.

name = "6h_WilliamsFractal_1dTrend_VolumeBreakout"
timeframe = "6h"
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

    # Get daily data for Williams fractals and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Williams fractals on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    # Williams fractals need 2 extra bars for confirmation (formed 2 days ago)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (1 day worth at 6h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > bearish fractal level + daily uptrend + volume spike
            if close[i] > bearish_fractal_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price < bullish fractal level + daily downtrend + volume spike
            elif close[i] < bullish_fractal_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < bullish fractal level OR trend reversal
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > bearish fractal level OR trend reversal
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals
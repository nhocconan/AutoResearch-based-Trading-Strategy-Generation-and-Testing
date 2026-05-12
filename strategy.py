#!/usr/bin/env python3
# 6h Williams Fractal Breakout with Volume Confirmation and Trend Filter
# Hypothesis: Williams fractals identify key support/resistance levels. 
# Breakout above bearish fractal (resistance) or below bullish fractal (support)
# with volume confirmation and 12h EMA50 trend filter captures momentum moves.
# Works in bull markets via upward breakouts and in bear markets via downward breakdowns.
# Williams fractals require 2-bar confirmation, so we use additional_delay_bars=2.

name = "6h_WilliamsFractal_Breakout_Volume_Trend"
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
    
    # === 12h Data for Williams Fractals and Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Fractals require 5-bar window (2 bars each side)
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_12h, low_12h)
    # Fractals need 2 additional bars for confirmation (total 3-bar delay from center)
    bearish_fractal_6h = align_htf_to_ltf(prices, df_12h, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_6h = align_htf_to_ltf(prices, df_12h, bullish_fractal, additional_delay_bars=2)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike detection (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready (50 EMA + fractal delays)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_6h[i]) or np.isnan(bearish_fractal_6h[i]) or 
            np.isnan(bullish_fractal_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above bearish fractal (resistance) with volume + uptrend
            if (close[i] > bearish_fractal_6h[i] and 
                bearish_fractal_6h[i] > 0 and  # Valid fractal level
                vol_spike[i] and 
                close[i] > ema_50_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below bullish fractal (support) with volume + downtrend
            elif (close[i] < bullish_fractal_6h[i] and 
                  bullish_fractal_6h[i] > 0 and  # Valid fractal level
                  vol_spike[i] and 
                  close[i] < ema_50_6h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls back below bullish fractal (support) or trend change
            if (close[i] < bullish_fractal_6h[i] and bullish_fractal_6h[i] > 0) or \
               close[i] < ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above bearish fractal (resistance) or trend change
            if (close[i] > bearish_fractal_6h[i] and bearish_fractal_6h[i] > 0) or \
               close[i] > ema_50_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
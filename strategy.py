#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d trend filter and volume confirmation
# Uses weekly Williams Fractals for structure (more reliable than daily), 1d EMA50 for trend filter,
# and volume spike for confirmation. Designed for 12-25 trades/year to minimize fee drag.
# Works in bull markets via upside breakouts at bullish fractals and in bear markets via downside breakdowns at bearish fractals.
# The 1d EMA50 provides a stable trend filter that adapts to changing market regimes.

name = "6h_WilliamsFractal_1dEMA50_VolumeSpike_TrendFilter"
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
    
    # Get 1w data for Williams Fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Williams Fractals on weekly data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal = np.full(len(high_1w), np.nan)
    bullish_fractal = np.full(len(low_1w), np.nan)
    
    for i in range(2, len(high_1w)-2):
        if (high_1w[i-2] < high_1w[i-1] and 
            high_1w[i] < high_1w[i-1] and
            high_1w[i-3] < high_1w[i-1] and
            high_1w[i+1] < high_1w[i-1]):
            bearish_fractal[i] = high_1w[i-1]
        
        if (low_1w[i-2] > low_1w[i-1] and 
            low_1w[i] > low_1w[i-1] and
            low_1w[i-3] > low_1w[i-1] and
            low_1w[i+1] > low_1w[i-1]):
            bullish_fractal[i] = low_1w[i-1]
    
    # Shift to use prior completed 1w bar levels (2-bar delay for fractal confirmation)
    bearish_fractal_shifted = np.roll(bearish_fractal, 1)
    bullish_fractal_shifted = np.roll(bullish_fractal, 1)
    bearish_fractal_shifted[0] = np.nan
    bullish_fractal_shifted[0] = np.nan
    
    # Align Williams Fractals to 6h timeframe with additional 2-bar delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal_shifted, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal_shifted, additional_delay_bars=2)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter from prior completed 1d bar
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_shifted = np.roll(ema50_1d, 1)
    ema50_1d_shifted[0] = np.nan
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above bullish fractal AND 1d EMA50 uptrend AND volume spike
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below bearish fractal AND 1d EMA50 downtrend AND volume spike
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below bullish fractal OR below 1d EMA50
            if close[i] < bullish_fractal_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above bearish fractal OR above 1d EMA50
            if close[i] > bearish_fractal_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
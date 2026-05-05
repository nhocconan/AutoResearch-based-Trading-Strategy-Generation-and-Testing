#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above the most recent bullish Williams Fractal (from completed 1d candles) AND price > 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below the most recent bearish Williams Fractal (from completed 1d candles) AND price < 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above the 1d EMA34 OR volume drops below average
# Williams Fractals require 2-bar confirmation after the center bar (additional_delay_bars=2)
# Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_WilliamsFractal_Breakout_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Fractals and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (bearish: high[2] is highest of 5, bullish: low[2] is lowest of 5)
    # Using 5-bar window: [i-2, i-1, i, i+1, i+2] where i is the center
    n_1d = len(high_1d)
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        if (high_1d[i] >= high_1d[i-1] and high_1d[i] >= high_1d[i-2] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]  # Bearish fractal at high[i]
        if (low_1d[i] <= low_1d[i-1] and low_1d[i] <= low_1d[i-2] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]   # Bullish fractal at low[i]
    
    # Williams Fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above bullish Williams Fractal, above 1d EMA34, volume confirmation, in session
            if close[i] > bullish_fractal_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below bearish Williams Fractal, below 1d EMA34, volume confirmation, in session
            elif close[i] < bearish_fractal_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 1d EMA34 OR volume drops below average
            if close[i] < ema34_1d_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 1d EMA34 OR volume drops below average
            if close[i] > ema34_1d_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Fractal breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d bullish Williams fractal AND 1w EMA34 > EMA34 previous (uptrend) AND volume > 1.8 * avg_volume(20) on 4h
# Short when price breaks below 1d bearish Williams fractal AND 1w EMA34 < EMA34 previous (downtrend) AND volume > 1.8 * avg_volume(20) on 4h
# Exit when price retests the 1d Williams fractal midpoint (average of bullish/bearish fractal levels)
# Uses discrete sizing 0.25 to minimize fee churn while maintaining edge
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams Fractals identify significant swing points with built-in confirmation delay
# 1w EMA34 provides responsive weekly trend filter with less lag than EMA50
# Volume confirmation validates breakout strength while limiting false signals
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "4h_WilliamsFractal_1wEMA34_Trend_Volume"
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
    
    # Get 1d data ONCE before loop for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 completed daily bars for Williams Fractals
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals (5-bar window: 2 left, center, 2 right)
    # Bullish fractal: low[center] < low[center-2] and low[center] < low[center-1] and low[center] < low[center+1] and low[center] < low[center+2]
    # Bearish fractal: high[center] > high[center-2] and high[center] > high[center-1] and high[center] > high[center+1] and high[center] > high[center+2]
    n_1d = len(high_1d)
    bullish_fractal = np.full(n_1d, np.nan)
    bearish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bullish fractal: lowest low in 5-bar window
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        # Bearish fractal: highest high in 5-bar window
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # Align 1d Williams Fractals to 4h timeframe (requires 2 extra bars for confirmation)
    bullish_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Calculate midpoint for exit
    midpoint = (bullish_aligned + bearish_aligned) / 2.0
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 completed weekly bars for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or np.isnan(midpoint[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d bullish Williams fractal, 1w EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (close[i] > bullish_aligned[i] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d bearish Williams fractal, 1w EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (close[i] < bearish_aligned[i] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the 1d Williams fractal midpoint
            if close[i] <= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the 1d Williams fractal midpoint
            if close[i] >= midpoint[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
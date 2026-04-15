#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h Williams Fractal Breakout with Volume Confirmation and Trend Filter
# Uses daily Williams Fractals for key support/resistance levels. Breaks above/below recent fractals
# are traded only with volume confirmation and 12h EMA trend alignment. Works in trending markets.
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals (5-bar window)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    for i in range(2, len(high_1d) - 2):
        # Bearish fractal: high[i] is highest among 5 bars
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: low[i] is lowest among 5 bars
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Align fractals to 4h with 2-bar delay for confirmation
    bearish_fractal_float = bearish_fractal.astype(float)
    bullish_fractal_float = bullish_fractal.astype(float)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_float, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_float, additional_delay_bars=2)
    
    # Get most recent fractal levels (carry forward last value)
    def get_last_fractal(arr):
        last_val = np.nan
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if not np.isnan(arr[i]):
                last_val = arr[i]
            result[i] = last_val
        return result
    
    last_bearish = get_last_fractal(bearish_fractal_aligned)
    last_bullish = get_last_fractal(bullish_fractal_aligned)
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(21)
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(last_bullish[i]) or np.isnan(last_bearish[i]) or
            np.isnan(ema_12h_aligned[i])):
            continue
        
        # Long entry: price above recent bullish fractal + volume confirmation + price > 12h EMA
        if (close[i] > last_bullish[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_12h_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below recent bearish fractal + volume confirmation + price < 12h EMA
        elif (close[i] < last_bearish[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_12h_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse fractal break or trend change
        elif position == 1 and (close[i] < last_bearish[i] or close[i] < ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > last_bullish[i] or close[i] > ema_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Williams_Fractal_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0
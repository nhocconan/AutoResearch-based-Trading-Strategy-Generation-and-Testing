#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal breakout with 1d trend filter and volume confirmation
# Uses Williams fractals from 1d to identify potential reversal points. Enters on breakout
# above bearish fractal or below bullish fractal with volume confirmation and 1d EMA trend filter.
# Works in both bull and bear markets by trading breakouts in the direction of the 1d trend.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Williams fractals and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals (5-bar pattern)
    # Bearish fractal: high[n-2] is highest of high[n-4:n+1]
    # Bullish fractal: low[n-2] is lowest of low[n-4:n+1]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] == np.max(high_1d[i-2:i+3]) and 
            high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        
        if (low_1d[i] == np.min(low_1d[i-2:i+3]) and 
            low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align fractals and EMA to 12h timeframe with 2-bar delay for fractal confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            continue
        
        # Long entry: price breaks above bearish fractal + volume confirmation + price > EMA34 (uptrend)
        if (close[i] > bearish_fractal_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
            close[i] > ema_34_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below bullish fractal + volume confirmation + price < EMA34 (downtrend)
        elif (close[i] < bullish_fractal_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-10):i+1]) and
              close[i] < ema_34_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite fractal breakout or trend reversal
        elif position == 1 and (close[i] < bullish_fractal_aligned[i] or close[i] < ema_34_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > bearish_fractal_aligned[i] or close[i] > ema_34_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Fractal_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
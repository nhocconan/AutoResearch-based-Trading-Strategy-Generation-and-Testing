#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Fractal for reversal points + 1d EMA trend filter + volume confirmation.
# Long when bullish fractal forms above 1d EMA34 and volume > 1.3x average.
# Short when bearish fractal forms below 1d EMA34 and volume > 1.3x average.
# Williams Fractal requires 2-bar confirmation, so we use additional_delay_bars=2.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.
# Works in bull (breakouts from fractal points) and bear (rejections at fractal points).

name = "12h_WilliamsFractal_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractal and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractal: 5-point pattern (high[low,low,high,high] or low[high,high,low,low])
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-3] < high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-3] > low[n-1] < low[n+1]
    n_1d = len(high_1d)
    bearish_fractal = np.zeros(n_1d, dtype=bool)
    bullish_fractal = np.zeros(n_1d, dtype=bool)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal: middle high is highest
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: middle low is lowest
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = True
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Williams Fractal needs 2-bar confirmation after the center bar
    bearish_fractal_confirmed = bearish_fractal.astype(float)
    bullish_fractal_confirmed = bullish_fractal.astype(float)
    
    # Align 1d indicators to 12h
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal_confirmed, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal_confirmed, additional_delay_bars=2
    )
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish fractal forms above EMA34 and volume spike
            if (bullish_fractal_aligned[i] and
                low[i] > ema_34_aligned[i] and  # Ensure price is above EMA
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: bearish fractal forms below EMA34 and volume spike
            elif (bearish_fractal_aligned[i] and
                  high[i] < ema_34_aligned[i] and  # Ensure price is below EMA
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: bearish fractal forms or price crosses below EMA34
            if (bearish_fractal_aligned[i] or
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish fractal forms or price crosses above EMA34
            if (bullish_fractal_aligned[i] or
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
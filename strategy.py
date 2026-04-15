#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d Williams Fractal + volume confirmation
# Uses 12h price channel breakouts for trend capture, 1d Williams Fractal for trend exhaustion signals,
# and volume to confirm breakout strength. Works in both bull and bear by
# only taking breakouts when fractals confirm trend continuation.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1d data for Williams Fractal calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate Williams Fractals on 1d (requires 5 candles: 2 left, center, 2 right)
    # Bearish fractal: high[i] is highest among [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest among [i-2, i-1, i, i+1, i+2]
    n1d = len(high_1d)
    bearish_fractal = np.zeros(n1d, dtype=bool)
    bullish_fractal = np.zeros(n1d, dtype=bool)
    
    for i in range(2, n1d - 2):
        # Bearish fractal: current high is highest in window
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: current low is lowest in window
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = True
    
    # Williams Fractal needs 2 extra 1d bars for confirmation (center bar + 2 right bars)
    bearish_fractal_confirmed = np.zeros(n1d, dtype=bool)
    bullish_fractal_confirmed = np.zeros(n1d, dtype=bool)
    for i in range(4, n1d):  # Start from index 4 to allow for 2-bar confirmation
        if bearish_fractal[i-2]:
            bearish_fractal_confirmed[i] = True
        if bullish_fractal[i-2]:
            bullish_fractal_confirmed[i] = True
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal_confirmed.astype(float), additional_delay_bars=0)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal_confirmed.astype(float), additional_delay_bars=0)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + no bearish fractal (no trend exhaustion)
        if (close[i] > donch_high_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            bearish_fractal_aligned[i] == 0 and  # No bearish fractal = no sell signal
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + no bullish fractal (no trend exhaustion)
        elif (close[i] < donch_low_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              bullish_fractal_aligned[i] == 0 and  # No bullish fractal = no buy signal
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or fractal appears (trend exhaustion)
        elif position == 1 and (close[i] < donch_low_12h_aligned[i] or bearish_fractal_aligned[i] == 1):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_12h_aligned[i] or bullish_fractal_aligned[i] == 1):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Fractal_Volume"
timeframe = "12h"
leverage = 1.0
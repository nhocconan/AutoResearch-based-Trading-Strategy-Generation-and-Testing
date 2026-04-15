#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and ATR stoploss
# Uses 20-period Donchian channels on 12h timeframe. Breakouts above upper band or below lower band
# are traded when confirmed by volume > 1.5x median volume of last 20 bars.
# Includes ATR-based stoploss to limit downside. Works in both bull and bear markets by
# capturing breakouts in either direction. Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR calculation (using daily data for more stable volatility)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR (14-period) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 periods
    upper_band = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_band = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_12h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_12h, lower_band)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(20, n):  # Start from 20 to ensure we have enough data for indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above upper Donchian band + volume confirmation
        if (close[i] > upper_band_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below lower Donchian band + volume confirmation
        elif (close[i] < lower_band_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: ATR-based stoploss or reverse breakout
        elif position == 1 and (close[i] < upper_band_aligned[i] - 2.0 * atr_1d_aligned[i] or
                                close[i] < lower_band_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > lower_band_aligned[i] + 2.0 * atr_1d_aligned[i] or
                                 close[i] > upper_band_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_Volume_ATR"
timeframe = "12h"
leverage = 1.0
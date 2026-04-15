#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d ATR volatility filter
# Uses 4-hour Donchian channel (20-period) for breakout signals, confirmed by volume spike
# and filtered by 1-day ATR-based volatility regime (only trade when volatility is elevated).
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-day ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period average ATR) to detect volatility expansion
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / (atr_ma_50 + 1e-10)
    
    # Align 1d indicators to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume confirmation + volatility expansion
        if (close[i] > donchian_high[i] and
            volume[i] > 2.0 * np.median(volume[max(0, i-10):i+1]) and
            atr_ratio_aligned[i] > 1.2 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume confirmation + volatility expansion
        elif (close[i] < donchian_low[i] and
              volume[i] > 2.0 * np.median(volume[max(0, i-10):i+1]) and
              atr_ratio_aligned[i] > 1.2 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite Donchian breakout or volatility contraction
        elif position == 1 and (close[i] < donchian_low[i] or atr_ratio_aligned[i] < 0.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donchian_high[i] or atr_ratio_aligned[i] < 0.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0
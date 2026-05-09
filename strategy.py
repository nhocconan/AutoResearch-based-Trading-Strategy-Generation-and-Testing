#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 1d volatility filter and volume confirmation
# Long when price breaks above 4h Donchian upper band (20) AND 1d ATR ratio indicates low volatility AND volume spike
# Short when price breaks below 4h Donchian lower band (20) AND 1d ATR ratio indicates low volatility AND volume spike
# Exit when price returns to 4h Donchian midpoint or volatility increases
# Uses volatility filter to avoid whipsaws in high volatility regimes, targeting 20-40 trades/year

name = "4h_Donchian_VolFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR for volatility filter (ATR ratio)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First value
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10) and ATR(30) for volatility ratio
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ratio = atr_10 / atr_30  # Low when < 1 (volatility contracting)
    vol_filter = atr_ratio < 1.0  # Volatility contracting or stable
    
    # Align daily volatility filter to 4h timeframe
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # Calculate 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_max
    donchian_lower = low_min
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume spike: current volume > 2x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Donchian and volatility
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(vol_filter_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + vol filter + volume spike
            if (close[i] > donchian_upper[i] and 
                vol_filter_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower + vol filter + volume spike
            elif (close[i] < donchian_lower[i] and 
                  vol_filter_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR volatility increases
            if (close[i] < donchian_mid[i]) or (not vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR volatility increases
            if (close[i] > donchian_mid[i]) or (not vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
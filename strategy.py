#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channel on daily data
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h (properly delayed for completed 1d bar)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_12h = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # wait for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or
            np.isnan(low_20_aligned[i]) or
            np.isnan(atr_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 30-period average (avoid low volatility chop)
        atr_ma_1d = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
        atr_ma_12h = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
        vol_filter = atr_1d[i] > atr_ma_1d[i] if not np.isnan(atr_ma_1d[i]) else False
        
        if position == 0:
            # Long: price breaks above 20-day high with volume and volatility filter
            if close[i] > high_20_aligned[i-1] and volume_spike_12h[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume and volatility filter
            elif close[i] < low_20_aligned[i-1] and volume_spike_12h[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-day low
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-day high
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1d_Volume_VolatilityFilter"
timeframe = "12h"
leverage = 1.0
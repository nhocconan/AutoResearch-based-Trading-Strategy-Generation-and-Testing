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
    
    # Get 12h data for Donchian channel (price channel)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h Donchian(20) channel
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donch_high_20 = high_12h_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_12h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # ATR(14) for volatility filter and stop
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high with volume
            if close[i] > donch_high_20_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low with volume
            elif close[i] < donch_low_20_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low OR ATR-based stop
            if close[i] < donch_low_20_aligned[i] or close[i] < (high[max(0, i-1)] - 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high OR ATR-based stop
            if close[i] > donch_high_20_aligned[i] or close[i] > (low[max(0, i-1)] + 1.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12h_Volume_ATR"
timeframe = "4h"
leverage = 1.0
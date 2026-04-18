#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation.
# Donchian breakout captures breakouts in trending markets.
# 1d ATR filter ensures we trade only when volatility is expanding (ATR > 1d ATR mean).
# Volume confirmation (>1.5x 20-period average) validates breakout strength.
# Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band).
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_Donchian20_1dATRFilter_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period ATR for 1d
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_mean_1d = pd.Series(atr_14_1d).rolling(window=30, min_periods=30).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_mean_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_mean_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_conf = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(atr_mean_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR > 1d ATR mean (expanding volatility)
        vol_filter = atr_14_1d_aligned[i] > atr_mean_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band + volume + volatility
            if close[i] > high_20[i] and volume_conf[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + volume + volatility
            elif close[i] < low_20[i] and volume_conf[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
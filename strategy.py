#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR-based volatility filter and volume confirmation
- Uses Donchian(20) for breakout detection: long on upper band break, short on lower band break
- 1d ATR(14) filter: only trade when ATR > 1.5x 50-period average (high volatility regimes)
- Volume confirmation: > 2.0x 20-period average volume to filter false breakouts
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by capturing volatility expansion breakouts
- ATR filter avoids low-volatility choppy markets where breakouts fail
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower bands (20-period high/low)
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14_1d > (1.5 * atr_ma_50_1d)
    
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter.astype(float))
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for ATR MA and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr_filter_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band with high volatility and volume
            long_breakout = (close[i] > donchian_upper_aligned[i] and 
                           atr_filter_aligned[i] > 0.5 and
                           volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian lower band with high volatility and volume
            short_breakout = (close[i] < donchian_lower_aligned[i] and 
                            atr_filter_aligned[i] > 0.5 and
                            volume[i] > 2.0 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or volatility contraction
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower band or volatility contracts
                if (close[i] < donchian_lower_aligned[i] or 
                    atr_filter_aligned[i] <= 0.5):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian upper band or volatility contracts
                if (close[i] > donchian_upper_aligned[i] or 
                    atr_filter_aligned[i] <= 0.5):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0
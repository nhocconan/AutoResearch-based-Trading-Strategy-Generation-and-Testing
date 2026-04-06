#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with daily ATR filter and volume confirmation
# Long when price breaks above Donchian high + daily ATR rising + volume > 1.5x average
# Short when price breaks below Donchian low + daily ATR rising + volume > 1.5x average
# Exit when price crosses Donchian midpoint or ATR falls below threshold
# Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# Works in trending markets by filtering breakouts with volatility expansion

name = "4h_donchian_atr_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR (14-period)
    atr = np.zeros_like(tr)
    atr[13] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR slope (rising if current > previous)
    atr_rising = np.zeros_like(atr, dtype=bool)
    atr_rising[1:] = atr[1:] > atr[:-1]
    
    # Align ATR and rising flag to 4h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_aligned[i]) or np.isnan(atr_rising_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses midpoint or ATR not rising
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or atr_rising_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or atr_rising_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volatility expansion and volume
            # Bullish breakout: price above Donchian high + ATR rising + volume
            if (close[i] > donch_high[i] and 
                atr_rising_aligned[i] > 0.5 and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + ATR rising + volume
            elif (close[i] < donch_low[i] and 
                  atr_rising_aligned[i] > 0.5 and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals
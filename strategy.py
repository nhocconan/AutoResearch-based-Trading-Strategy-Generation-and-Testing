#!/usr/bin/env python3
"""
6h Volume Spike + Keltner Channel Breakout
Hypothesis: Volume spikes indicate institutional interest. Keltner Channel (ATR-based) provides
dynamic support/resistance. Long when price breaks above upper band with volume spike;
short when breaks below lower band with volume spike. Works in both bull (breakouts)
and bear (breakdowns). Uses 1d ATR for channel calculation to reduce noise.
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14419_6h_volume_spike_keltner_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ATR (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR for Keltner Channel (20-period)
    atr_period = 20
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_std = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_threshold = vol_ma + (2.0 * vol_std)  # 2 sigma above mean
    
    # Keltner Channel on 6h using 1d ATR
    ma_period = 20
    ma = pd.Series(close).rolling(window=ma_period, min_periods=ma_period).mean().values
    atr_6h = pd.Series(atr_1d).rolling(window=6, min_periods=1).mean().values  # Approximate 6h ATR from 1d
    keltner_upper = ma + (2.0 * atr_6h)
    keltner_lower = ma - (2.0 * atr_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(atr_period, ma_period) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ma[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(vol_ma[i]) or np.isnan(vol_std[i]) or np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > vol_threshold[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price below middle line OR stoploss
            if (close[i] <= ma[i] or close[i] <= entry_price - 2.5 * atr_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above middle line OR stoploss
            if (close[i] >= ma[i] or close[i] >= entry_price + 2.5 * atr_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price outside channel + volume spike
            long_setup = (close[i] > keltner_upper[i]) and vol_spike
            short_setup = (close[i] < keltner_lower[i]) and vol_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
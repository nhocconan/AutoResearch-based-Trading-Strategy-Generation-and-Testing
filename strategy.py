# SPDX-FileCopyrightText: 2025 Alpaca Wong
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation.
In trending markets (1d ATR rising), buy breakouts above 4h Donchian high; sell breakdowns below 4h Donchian low.
1d ATR acts as a volatility filter to avoid ranging markets. Volume confirms breakout strength.
Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d ATR rising: current ATR > ATR 3 periods ago
    atr_rising = np.zeros_like(atr_14, dtype=bool)
    atr_rising[3:] = atr_14[3:] > atr_14[:-3]
    atr_rising_aligned = align_htf_to_ltf(prices, df_1d, atr_rising)
    
    # 4h Donchian channel (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(prices['volume'].values, vol_ma_20, out=np.zeros_like(prices['volume'].values), where=vol_ma_20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr_rising_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        atr_ok = atr_rising_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high + ATR rising + volume spike
            if (price_close > donch_high[i] and 
                atr_ok and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 4h Donchian low + ATR rising + volume spike
            elif (price_close < donch_low[i] and 
                  atr_ok and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: opposite breakout (reversion to mean)
            if position == 1 and price_close < donch_low[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_1dATR_Volume"
timeframe = "4h"
leverage = 1.0
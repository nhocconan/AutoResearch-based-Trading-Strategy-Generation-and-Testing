#!/usr/bin/env python3
"""
4h Donchian(20) breakout with volume confirmation and ATR stoploss
Hypothesis: Donchian breakouts on 4h timeframe capture medium-term trends with clear support/resistance levels.
Volume confirmation filters false breakouts, ATR stoploss manages risk. Works in bull (breakout continuation)
and bear (breakdown continuation). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate rolling max/min for Donchian
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe
    donch_high_4h = align_htf_to_ltf(prices, df_1d, high_max)
    donch_low_4h = align_htf_to_ltf(prices, df_1d, low_min)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and ATR
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: stoploss or breakdown below Donchian low
            if (close[i] <= entry_price - 2.5 * atr[i] or
                close[i] <= donch_low_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: stoploss or breakout above Donchian high
            if (close[i] >= entry_price + 2.5 * atr[i] or
                close[i] >= donch_high_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume confirmation
            breakout_long = (close[i] > donch_high_4h[i] and
                           volume[i] > vol_ema[i] * 1.5)
            breakout_short = (close[i] < donch_low_4h[i] and
                            volume[i] > vol_ema[i] * 1.5)
            
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Daily Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts on 6h capture momentum, daily volume confirms breakout strength,
and ATR stoploss limits drawdown. Designed for low trade frequency (target 50-150 total over 4 years)
to minimize fee decay. Works in both bull and bear by filtering breakouts with volume and
ATR-based risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 20-period volume MA on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(vol_20d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter: current volume > 1.5x 20-day average volume
        volume_filter = volume[i] > vol_20d_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if bull_breakout and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals
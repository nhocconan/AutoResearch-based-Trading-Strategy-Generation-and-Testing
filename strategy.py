#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + Volume Confirmation + ATR Stop
Hypothesis: Donchian breakouts capture institutional momentum. Volume confirmation ensures breakout strength.
Works in bull (breakouts with volume) and bear (breakdowns with volume). Target: 75-250 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period ATR
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 19 + atr[i-1]) / 20
    
    # 20-period Donchian channels (using previous day's data to avoid look-ahead)
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])  # previous 20 days, not including today
        low_20[i] = np.min(low[i-20:i])
    
    # 20-period volume average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit
    
    # Start from index 20
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below 20-day low OR stoploss hit
            if (close[i] < low_20[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price rises above 20-day high OR stoploss hit
            if (close[i] > high_20[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Look for entries with minimum 5 bars since last exit
            if bars_since_exit >= 5:
                # Long: break above 20-day high with volume
                if close[i] > high_20[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: break below 20-day low with volume
                elif close[i] < low_20[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals
</lyht>
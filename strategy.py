#!/usr/bin/env python3
"""
4h Donchian(20) breakout with volume confirmation and ATR stop
Hypothesis: Price breaking 20-period high/low with volume confirmation captures
institutional breakout momentum. Works in bull (breakouts) and bear (breakdowns).
ATR stop limits drawdown. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_vol_stop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
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
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 20-period Donchian channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donch_high[i] = np.max(high[i-20:i])
            donch_low[i] = np.min(low[i-20:i])
    
    # 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    if len(vol_1d) >= 20:
        for i in range(20, len(vol_1d)):
            vol_ma_1d[i] = np.mean(vol_1d[i-20:i])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0.0
    bars_since_exit = 0
    
    start = 20
    
    for i in range(start, n):
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter: current volume > 1.5x 1d average
        vol_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        if position == 1:
            # Exit: close below Donchian low OR 2*ATR stop
            if close[i] < donch_low[i] or close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.25
            bars_since_exit += 1
        elif position == -1:
            # Exit: close above Donchian high OR 2*ATR stop
            if close[i] > donch_high[i] or close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.25
            bars_since_exit += 1
        else:
            # Minimum 8 bars between trades
            if bars_since_exit >= 8:
                # Long: break above Donchian high with volume
                if close[i] > donch_high[i] and vol_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: break below Donchian low with volume
                elif close[i] < donch_low[i] and vol_filter:
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
#!/usr/bin/env python3
"""
12h_1d_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Daily Donchian channel breakouts on 12h timeframe with volume confirmation and ATR stoploss.
Works in both bull and bear markets by capturing breakouts from volatility contractions.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Donchian channels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Calculate daily Donchian channels (20-period)
    high_20 = np.full_like(high_daily, np.nan)
    low_20 = np.full_like(low_daily, np.nan)
    for i in range(len(high_daily)):
        if i >= 19:
            high_20[i] = np.max(high_daily[i-19:i+1])
            low_20[i] = np.min(low_daily[i-19:i+1])
    
    # Align daily Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_daily, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_daily, low_20)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 24-period average (more selective)
    volume_avg = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        elif i > 0:
            volume_avg[i] = np.mean(volume[:i])
    
    volume_filter = volume > (1.8 * volume_avg)
    
    # ATR for stoploss (14-period)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            if i > 0:
                atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(24, n):
        # Skip if NaN in critical values
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_avg[i])):
            if position != 0:
                # Check stoploss
                if position == 1 and low[i] <= entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and high[i] >= entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * 0.25
            continue
        
        price = close[i]
        high_20 = high_20_aligned[i]
        low_20 = low_20_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long breakout: price breaks above 20-day high with volume
            if price > high_20 and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price breaks below 20-day low with volume
            elif price < low_20 and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long: trail stop or exit on mean reversion to midpoint
            midpoint = (high_20 + low_20) / 2.0
            # Stoploss
            if low[i] <= entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price returns to midpoint (mean reversion)
            elif price <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: trail stop or exit on mean reversion to midpoint
            midpoint = (high_20 + low_20) / 2.0
            # Stoploss
            if high[i] >= entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price returns to midpoint (mean reversion)
            elif price >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Donchian20_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0
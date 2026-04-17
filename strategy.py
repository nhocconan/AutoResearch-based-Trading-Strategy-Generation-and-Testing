#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
Enters long when price breaks above Donchian upper band with volume > 1.5x average and ADX > 25.
Enters short when price breaks below Donchian lower band with volume > 1.5x average and ADX > 25.
Exits when price crosses the Donchian middle (20-period midpoint). Uses ADX to filter range markets.
Designed to work in both bull and bear markets by following established trends with volume confirmation.
Target: 20-50 trades/year to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # ADX calculation (14-period)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    period = 14
    # Initial values
    atr[period] = np.mean(tr[1:period+1])
    plus_di[period] = 100 * np.mean(plus_dm[1:period+1]) / atr[period]
    minus_di[period] = 100 * np.mean(minus_dm[1:period+1]) / atr[period]
    
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / period * 100 / atr[i] if atr[i] != 0 else 0
        minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / period * 100 / atr[i] if atr[i] != 0 else 0
        dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
    
    # ADX is smoothed DX
    adx[period*2] = np.mean(dx[period+1:period*2+1])
    for i in range(period*2+1, n):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # enough for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume and ADX > 25
            if close[i] > highest_high[i] and volume_filter[i] and adx[i] > 25:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower with volume and ADX > 25
            elif close[i] < lowest_low[i] and volume_filter[i] and adx[i] > 25:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Volume_ADX25"
timeframe = "4h"
leverage = 1.0
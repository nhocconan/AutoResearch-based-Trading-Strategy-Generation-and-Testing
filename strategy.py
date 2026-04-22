#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# This strategy trades breakouts of 20-period Donchian channels with volatility filtering
# to avoid false breakouts in low-volatility environments. It uses 1d ATR to ensure
# sufficient volatility for meaningful breakouts and volume confirmation for institutional
# participation. Works in both bull and bear markets by following breakout direction.
# Uses discrete position sizing (0.30) to balance return and minimize transaction costs.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ATR calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is NaN
    atr_14 = np.zeros(len(tr))
    for i in range(len(tr)):
        if i < 14:
            atr_14[i] = np.nan
        elif i == 14:
            atr_14[i] = np.nanmean(tr[1:15])  # Average of first 14 TR values (skip first NaN)
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-period average
    vol_avg_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to ensure Donchian data
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: require ATR > 0.5 * price (adaptive threshold)
        if atr_14_aligned[i] <= 0.5 * close[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above upper Donchian + volatility + volume spike
            if close[i] > highest_20[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below lower Donchian + volatility + volume spike
            elif close[i] < lowest_20[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price crosses opposite Donchian band
            if position == 1:
                # Exit long: Close below lower Donchian
                if close[i] < lowest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                # Exit short: Close above upper Donchian
                if close[i] > highest_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day ATR volatility filter and volume spike.
Long when price breaks above 20-period high with ATR(14) > 1.5x ATR(50) and volume spike.
Short when price breaks below 20-period low with ATR(14) > 1.5x ATR(50) and volume spike.
Exit when price retests the 20-period midpoint.
ATR volatility filter ensures trades occur during high volatility periods, reducing false breakouts.
Volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations. Works in both bull and bear markets by following price channels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1-day ATR(14) and ATR(50) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(14)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR(50)
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align to 4h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i]) or
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR volatility filter: ATR(14) > 1.5x ATR(50)
        vol_filter = atr14_1d_aligned[i] > 1.5 * atr50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-period high with volatility filter and volume spike
            if (close[i] > high_20[i] and vol_filter and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low with volatility filter and volume spike
            elif (close[i] < low_20[i] and vol_filter and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price retests 20-period midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < mid_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midpoint
                if close[i] > mid_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dATR_Vol_Filter_Volume"
timeframe = "4h"
leverage = 1.0
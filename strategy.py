#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian Breakout with 1-day ATR filter and volume confirmation.
Trades breakouts of 12-hour Donchian channels only when daily ATR volatility is elevated
and volume confirms institutional interest. Uses 1-day ATR as a regime filter to avoid
whipsaw in low volatility environments. Designed for low trade frequency (12-37/year)
to minimize fee drift and work in both bull and bear markets by aligning with volatility regime.
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
    
    # Load daily data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily ATR for volatility regime filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 12-hour Donchian channel (20-period)
    high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(atr_14_12h[i]) or np.isnan(high_12h[i]) or 
            np.isnan(low_12h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma_50 = pd.Series(atr_14_12h).rolling(window=50, min_periods=50).mean().values
        vol_regime = atr_14_12h[i] > atr_ma_50[i] if not np.isnan(atr_ma_50[i]) else False
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_regime and vol_spike:
            # Long: price breaks above 12h Donchian upper band
            if close[i] > high_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian lower band
            elif close[i] < low_12h[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to midpoint of Donchian channel
            exit_signal = False
            mid = (high_12h[i] + low_12h[i]) / 2
            
            if position == 1:
                # Exit long: price falls below midpoint
                if close[i] < mid:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price rises above midpoint
                if close[i] > mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Volume_Breakout"
timeframe = "12h"
leverage = 1.0
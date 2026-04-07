#!/usr/bin/env python3
"""
6h Donchian Breakout + 1D Trend Filter + Volume Confirmation
Breakout strategy using Donchian channels (20) on 6h, filtered by 1D EMA trend direction,
and confirmed by volume spike (>1.5x average). Works in both bull and bear markets
by only taking breakouts in the direction of the higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h Donchian Channel (20) ===
    # Calculate rolling max/min with proper min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 1D EMA Trend Filter ===
    # Get 1D data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate EMA21 on 1D close
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1D EMA
        # Uptrend: price above EMA21, Downtrend: price below EMA21
        trend_up = close[i] > ema_1d_aligned[i]
        trend_down = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low (contrarian exit)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (contrarian exit)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry logic: breakout in direction of 1D trend
            # Long: break above Donchian high in uptrend
            if close[i] > donchian_high[i] and trend_up:
                position = 1
                signals[i] = 0.25
            # Short: break below Donchian low in downtrend
            elif close[i] < donchian_low[i] and trend_down:
                position = -1
                signals[i] = -0.25
    
    return signals
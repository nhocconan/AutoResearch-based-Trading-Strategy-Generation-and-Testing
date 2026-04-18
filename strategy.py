#!/usr/bin/env python3
"""
12h_1D_PriceChannel_Breakout_Volume_Filter
Hypothesis: Use daily price channel (Donchian 20) for trend direction and 12H for entry with volume confirmation.
Long when price breaks above 20-day high with volume > 1.3x average during active session (08-20 UTC).
Short when price breaks below 20-day low with volume > 1.3x average during active session.
Fixed position size 0.25. Added volatility filter (ATR) to avoid chop.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via volatility regime filter and session timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-day Donchian channel: upper = max(high_20), lower = min(low_20)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align all daily data to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volatility filter: use ATR(20) to avoid choppy markets
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first day
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for Donchian and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.3 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Volatility filter: avoid extreme volatility (stop hunting)
        vol_ma_long = pd.Series(atr_20_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr_20_aligned[i] < vol_ma_long[i] * 2 if not np.isnan(vol_ma_long[i]) else False
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above 20-day high with volume and volatility filter during session
            if close[i] > high_20_aligned[i] and vol_confirm and vol_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with volume and volatility filter during session
            elif close[i] < low_20_aligned[i] and vol_confirm and vol_filter and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below 20-day high or volatility spike or outside session
            if close[i] < high_20_aligned[i] or not vol_filter or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above 20-day low or volatility spike or outside session
            if close[i] > low_20_aligned[i] or not vol_filter or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_PriceChannel_Breakout_Volume_Filter"
timeframe = "12h"
leverage = 1.0
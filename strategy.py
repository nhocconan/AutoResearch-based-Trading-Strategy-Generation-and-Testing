#!/usr/bin/env python3
"""
12h_1D_Trend_Confluence_V1
Hypothesis: Use 1D EMA34 for daily trend direction, 12H for entry with volume confirmation.
Long when price > daily EMA34 and breaks above 12H high of last 20 bars with volume > 1.3x average.
Short when price < daily EMA34 and breaks below 12H low of last 20 bars with volume > 1.3x average.
Only trade during active session (08-20 UTC). Fixed position size 0.25.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drain.
Works in bull/bear via daily EMA trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on daily close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12H Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price > daily EMA34 and breaks above 20-period high with volume
            if close[i] > ema_34_aligned[i] and close[i] > high_20[i] and vol_confirm and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price < daily EMA34 and breaks below 20-period low with volume
            elif close[i] < ema_34_aligned[i] and close[i] < low_20[i] and vol_confirm and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 20-period low or outside session
            if close[i] < low_20[i] or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 20-period high or outside session
            if close[i] > high_20[i] or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Trend_Confluence_V1"
timeframe = "12h"
leverage = 1.0
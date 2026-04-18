#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1S1_RangeFilter_V1
Hypothesis: Use 1D Camarilla R1/S1 levels with range filter (ATR-based) to trade breakouts only in trending markets, avoiding chop.
Long when price breaks above daily R1 with ATR(20) > 1.2 * ATR(50) during active session (08-20 UTC).
Short when price breaks below daily S1 with ATR(20) > 1.2 * ATR(50) during active session.
Fixed position size 0.25. Designed for 12-37 trades/year to avoid fee drag.
Works in bull/bear via trend filter and session timing.
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
    
    # Get daily data for Camarilla pivot levels and ATR
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first day uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # True Range for ATR
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first day
    
    # ATR(20) for trend filter and ATR(50) for long-term average
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align all daily data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for ATR(50)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_20_aligned[i]) or np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ATR(20) > 1.2 * ATR(50) indicates trending market
        trend_filter = atr_20_aligned[i] > 1.2 * atr_50_aligned[i]
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: price breaks above R1 with trend filter during session
            if close[i] > r1_aligned[i] and trend_filter and in_session:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with trend filter during session
            elif close[i] < s1_aligned[i] and trend_filter and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or trend filter fails or outside session
            if close[i] < r1_aligned[i] or not trend_filter or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or trend filter fails or outside session
            if close[i] > s1_aligned[i] or not trend_filter or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1D_Camarilla_R1S1_RangeFilter_V1"
timeframe = "12h"
leverage = 1.0
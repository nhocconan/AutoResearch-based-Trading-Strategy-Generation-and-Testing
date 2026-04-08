#!/usr/bin/env python3
# 12h_camarilla_pivot_daily_trend_volume_v1
# Hypothesis: Uses daily Camarilla pivot levels with volume confirmation and daily trend filter.
# Enters long when price breaks above R4 with volume spike and daily uptrend.
# Enters short when price breaks below S4 with volume spike and daily downtrend.
# Uses daily trend filter for multi-timeframe alignment. Designed for 12-37 trades/year to avoid fee drag.
# Camarilla levels derived from previous day's OHLC: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L) etc.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day (based on previous day's OHLC)
    # H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # H3 = C + 1.125*(H-L), L3 = C - 1.125*(H-L)
    # H6 = 1.5*H - 0.5*L, L6 = 1.5*L - 0.5*H
    
    hl_range = high_1d - low_1d
    h4 = close_1d + 1.5 * hl_range
    l4 = close_1d - 1.5 * hl_range
    h6 = 1.5 * high_1d - 0.5 * low_1d
    l6 = 1.5 * low_1d - 0.5 * high_1d
    
    # Align daily levels to 12h timeframe (previous day's levels available at open)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h6_aligned = align_htf_to_ltf(prices, df_1d, h6)
    l6_aligned = align_htf_to_ltf(prices, df_1d, l6)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: breakdown below L6 or daily trend failure
            if close[i] < l6_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above H6 or daily trend failure
            if close[i] > h6_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: break above H4 with volume spike and daily uptrend
                if close[i] > h4_aligned[i] and daily_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short entry: break below L4 with volume spike and daily downtrend
                elif close[i] < l4_aligned[i] and daily_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals
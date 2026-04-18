#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_With_1D_ADX_Filter
Hypothesis: On 1h timeframe, go long when price breaks above R1 (3.82/7) of daily Camarilla pivot with ADX>25 (trending), short when breaks below S1 with ADX>25. Uses 1d ADX for regime filter and 1h price for entry timing. Targets 15-30 trades/year with position size 0.20 to avoid overtrading. Works in bull/bear by only taking trend-aligned breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1D data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    # R1 = Close + (High - Low) * 1.0833
    # S1 = Close - (High - Low) * 1.0833
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.0833
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.0833
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/14)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(tr14 > 0, 100 * plus_dm14 / tr14, 0)
    minus_di = np.where(tr14 > 0, 100 * minus_dm14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    # Align to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    adx_1h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need ADX to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or 
            np.isnan(adx_1h[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending = adx_1h[i] > 25
        
        if position == 0:
            # Long entry: price breaks above R1 with ADX > 25
            if close[i] > r1_1h[i] and trending:
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 with ADX > 25
            elif close[i] < s1_1h[i] and trending:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below R1
            if close[i] < r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses back above S1
            if close[i] > s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_With_1D_ADX_Filter"
timeframe = "1h"
leverage = 1.0
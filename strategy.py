#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Volume_v2
Hypothesis: Uses weekly ADX (14) as trend filter and daily Camarilla pivot levels (H3/L3) for breakouts.
Trades only when price breaks H3/L3 with volume expansion and strong weekly trend (ADX > 25).
Exits when price returns to previous day's close.
Designed to work in both bull and bear markets by capturing breakouts with strong momentum.
Target: 12-37 trades per year (50-150 total over 4 years) to minimize fee drag.
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
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    hl_range = high_1d - low_1d
    H3 = close_1d + 1.125 * hl_range
    L3 = close_1d - 1.125 * hl_range
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14) on weekly data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[1]
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        plus_dm_smooth[period-1] = np.mean(plus_dm[1:period]) if period > 1 else plus_dm[1]
        minus_dm_smooth[period-1] = np.mean(minus_dm[1:period]) if period > 1 else minus_dm[1]
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # Calculate ADX as smoothed DX
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1]) if (2*period-1) < len(dx) else 0
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align all signals to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_1w_aligned[i] > 25
        
        # Entry conditions: price breaks H3/L3 with volume expansion and trend filter
        long_entry = (high[i] > H3_aligned[i]) and volume_expansion[i] and strong_trend
        short_entry = (low[i] < L3_aligned[i]) and volume_expansion[i] and strong_trend
        
        # Exit conditions: return to previous day's close
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        exit_long = position == 1 and close[i] <= prev_close_aligned[i]
        exit_short = position == -1 and close[i] >= prev_close_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_Camarilla_Breakout_Volume_v2"
timeframe = "12h"
leverage = 1.0
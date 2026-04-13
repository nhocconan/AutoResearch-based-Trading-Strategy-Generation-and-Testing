#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Volume_Trend_v1
Hypothesis: Combines 4h Camarilla breakout with 12h volume confirmation and 12h trend filter (ADX > 25).
Enters long when price breaks above H3 with volume expansion and strong 12h trend.
Enters short when price breaks below L3 with volume expansion and strong 12h trend.
Exits when price returns to previous 4h close.
Designed for 4h timeframe to balance trade frequency and signal quality.
Target: 20-50 trades per year (80-200 total over 4 years) to minimize fee drag.
Works in both bull and bear markets by requiring strong trend alignment.
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
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels for previous 4h bar
    hl_range = high_4h - low_4h
    H3 = close_4h + 1.125 * hl_range
    L3 = close_4h - 1.125 * hl_range
    
    # Get 12h data for ADX trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ADX (14) on 12h data
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
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Calculate 20-period volume average on 12h
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all signals to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_4h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_4h, L3)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_12h_aligned[i] > 25
        
        # Volume confirmation: current 4h volume > 1.5x 12h volume MA
        volume_expansion = volume[i] > (vol_ma_20_12h_aligned[i] * 1.5)
        
        # Entry conditions: price breaks H3/L3 with volume expansion and trend filter
        long_entry = (high[i] > H3_aligned[i]) and volume_expansion and strong_trend
        short_entry = (low[i] < L3_aligned[i]) and volume_expansion and strong_trend
        
        # Exit conditions: return to previous 4h close
        prev_close_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
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

name = "4h_12h_Camarilla_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0
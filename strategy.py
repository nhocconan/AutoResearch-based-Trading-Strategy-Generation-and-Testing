#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_With_Volume_Filter_v2
Hypothesis: Use daily Camarilla R1/S1 levels for breakout detection with volume confirmation and ADX regime filter. Designed to capture strong momentum moves in both bull and bear markets while avoiding choppy conditions. Targets 20-40 trades/year with position size 0.25.
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
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + range_1d * 1.1 / 12
    camarilla_s1 = close_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for daily bar close)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate ADX for regime filter (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros(len(high))
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        
        atr[period-1] = np.mean(tr[1:period]) if period > 1 else tr[0]
        plus_dm_sum = np.sum(plus_dm[1:period])
        minus_dm_sum = np.sum(minus_dm[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - plus_dm_sum/period + plus_dm[i]
            minus_dm_sum = minus_dm_sum - minus_dm_sum/period + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / (atr[i] * period) if atr[i] != 0 else 0
        
        dx = np.zeros(len(high))
        adx = np.zeros(len(high))
        for i in range(2*period-1, len(high)):
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100 * di_diff / di_sum if di_sum != 0 else 0
        
        adx[2*period-1] = np.mean(dx[2*period-1:3*period-1]) if 2*period < len(high) else 0
        for i in range(3*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # ADX filter: only trade when trending (ADX > 25)
        trending = adx[i] > 25
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and trending
            if close[i] > r1_4h[i] and vol_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation and trending
            elif close[i] < s1_4h[i] and vol_confirmed and trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses back below S1
            if close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above R1
            if close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_With_Volume_Filter_v2"
timeframe = "4h"
leverage = 1.0
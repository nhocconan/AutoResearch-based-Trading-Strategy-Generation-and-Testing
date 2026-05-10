#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1dTrend_ADXFilter
Hypothesis: Breakouts at 1d Camarilla R1/S1 levels with volume confirmation and 1d trend (ADX > 25) capture directional moves in trending markets. Avoids choppy markets using ADX filter to reduce false signals and lower trade frequency. Designed for low trade frequency (<30/year) to minimize fee drift while maintaining edge in both bull and bear markets by following strong trends.
"""

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_ADXFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d data for Camarilla and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous 1d bar for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    range_1d = high_1d - low_1d
    s1 = close_1d - (range_1d * 1.08333)
    r1 = close_1d + (range_1d * 1.08333)
    
    # Align to 4h timeframe (wait for 1d bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # 1d ADX filter for trend strength (avoid choppy markets)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        def wilders_smoothing(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(arr[1:period]) / period
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        plus_di = 100 * wilders_smoothing(plus_dm, period) / wilders_smoothing(tr, period)
        minus_di = 100 * wilders_smoothing(minus_dm, period) / wilders_smoothing(tr, period)
        dx = np.zeros_like(high)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: volume > 2.0x 20-period average (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for trending market (ADX > 25)
        is_trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume confirmation + trending market
            if (close[i] > r1_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 + volume confirmation + trending market
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S1 (opposite side)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeFilter_Tight
Hypothesis: Camarilla pivot levels (R1, S1) from 1d timeframe act as key support/resistance.
Long when price breaks above R1 with volume confirmation; short when price breaks below S1 with volume confirmation.
Only trade in trending markets (ADX > 25) to avoid whipsaw in chop. Works in bull/bear by following momentum.
Uses tight entry conditions to limit trades (<50/year) and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1])
        minus_dm[i] = max(0, low[i-1] - low[i])
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(high)
    plus_dm_smooth = np.zeros_like(high)
    minus_dm_smooth = np.zeros_like(high)
    
    atr[period] = np.mean(tr[1:period+1])
    plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
    minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
    
    for i in range(period+1, len(high)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    adx = np.zeros_like(high)
    adx[2*period] = np.mean(dx[period+1:2*period+1])
    for i in range(2*period+1, len(high)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R1 = np.zeros_like(close_1d)
    S1 = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        range_ = high_1d[i-1] - low_1d[i-1]
        close_prev = close_1d[i-1]
        R1[i] = close_prev + range_ * 1.1 / 12
        S1[i] = close_prev - range_ * 1.1 / 12
    
    # ADX for trend filter
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4h price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        adx_val = adx_aligned[i]
        
        # Only trade in trending markets (ADX > 25)
        trending = adx_val > 25
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in trending market
            if price > r1 and vol_spike and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in trending market
            elif price < s1 and vol_spike and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below S1 or loses momentum (ADX < 20)
            if price < s1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or loses momentum (ADX < 20)
            if price > r1 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeFilter_Tight"
timeframe = "4h"
leverage = 1.0
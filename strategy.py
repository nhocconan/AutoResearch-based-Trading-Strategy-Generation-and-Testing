#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot R1/S1 Breakout with Volume Spike and ADX Trend Filter.
Long when price breaks above R1 with volume > 1.8x average and ADX > 25 (trending market).
Short when price breaks below S1 with volume > 1.8x average and ADX > 25.
Exit when price reverts to pivot point (PP) or ADX < 20 (range begins).
Uses 1d for Camarilla pivot and ADX calculation, 12h for price/volume.
Target: 80-120 total trades over 4 years (20-30/year) to balance edge and fee drag.
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
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1, PP)
    def calculate_camarilla(high, low, close):
        pp = (high + low + close) / 3.0
        r1 = close + (high - low) * 1.1 / 12.0
        s1 = close - (high - low) * 1.1 / 12.0
        return pp, r1, s1
    
    pp_1d = np.zeros_like(close_1d)
    r1_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        pp, r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pp_1d[i] = pp
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                dm_plus[i] = up_move
            else:
                dm_plus[i] = 0
            if down_move > up_move and down_move > 0:
                dm_minus[i] = down_move
            else:
                dm_minus[i] = 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        tr_period = np.zeros_like(high)
        dm_plus_period = np.zeros_like(high)
        dm_minus_period = np.zeros_like(high)
        
        tr_period[period] = np.mean(tr[1:period+1])
        dm_plus_period[period] = np.mean(dm_plus[1:period+1])
        dm_minus_period[period] = np.mean(dm_minus[1:period+1])
        
        for i in range(period+1, len(high)):
            tr_period[i] = (tr_period[i-1] * (period-1) + tr[i]) / period
            dm_plus_period[i] = (dm_plus_period[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_period[i] = (dm_minus_period[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(high)
        di_minus = np.zeros_like(high)
        for i in range(period, len(high)):
            if tr_period[i] > 0:
                di_plus[i] = 100 * dm_plus_period[i] / tr_period[i]
                di_minus[i] = 100 * dm_minus_period[i] / tr_period[i]
            else:
                di_plus[i] = 0
                di_minus[i] = 0
        
        # DX and ADX
        dx = np.zeros_like(high)
        for i in range(period, len(high)):
            if di_plus[i] + di_minus[i] > 0:
                dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
            else:
                dx[i] = 0
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        pp = pp_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        
        # Trend regime: ADX > 25 = trending (good for breakout)
        is_trending = adx_val > 25
        # Range regime: ADX < 20 = ranging (avoid false breakouts)
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long: price breaks above R1 with volume spike in trending market
            if price > r1 and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike in trending market
            elif price < s1 and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point OR trend ends (ranging begins)
            if price <= pp or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR trend ends (ranging begins)
            if price >= pp or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_ADXTrend"
timeframe = "12h"
leverage = 1.0
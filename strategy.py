#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla Pivot R3/S3 Breakout with Volume Spike and ADX Trend Filter.
Long when price breaks above R3 with volume > 1.5x average and ADX > 25 (trending).
Short when price breaks below S3 with volume > 1.5x average and ADX > 25 (trending).
Exit when price reverts to pivot point (PP) or ADX < 20 (trend weakens).
Uses 1d for Camarilla pivot calculation and ADX, 12h for price/volume.
Target: 50-150 total trades over 4 years (12-37/year). Uses tighter R3/S3 levels and trend filter to reduce false breakouts.
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
    
    # Calculate 1d Camarilla pivot levels (R3, S3, PP)
    def calculate_camarilla(high, low, close):
        pp = (high + low + close) / 3.0
        r3 = close + (high - low) * 1.1 / 4.0
        s3 = close - (high - low) * 1.1 / 4.0
        return pp, r3, s3
    
    pp_1d = np.zeros_like(close_1d)
    r3_1d = np.zeros_like(close_1d)
    s3_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        pp, r3, s3 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pp_1d[i] = pp
        r3_1d[i] = r3
        s3_1d[i] = s3
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        atr = np.zeros_like(close)
        dm_plus_smooth = np.zeros_like(close)
        dm_minus_smooth = np.zeros_like(close)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        dm_plus_smooth[period] = np.mean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(close)
        di_minus = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if atr[i] > 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
                if di_plus[i] + di_minus[i] > 0:
                    dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
                else:
                    dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])  # First ADX value
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        pp = pp_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        
        # Trend regime: ADX > 25 = strong trend (good for breakout)
        is_trending = adx_val > 25
        # Weak trend: ADX < 20 = trend weakening (exit signal)
        is_weak_trend = adx_val < 20
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and strong trend
            if price > r3 and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and strong trend
            elif price < s3 and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point OR trend weakens
            if price <= pp or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point OR trend weakens
            if price >= pp or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_VolumeSpike_ADXTrend"
timeframe = "12h"
leverage = 1.0
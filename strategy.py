#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above 20-period high with volume > 1.5x average and 1d ADX > 25.
Short when price breaks below 20-period low with volume > 1.5x average and 1d ADX > 25.
Exit when price reverts to 20-period midpoint or ADX < 20 (trend weakening).
Uses 1d for ADX calculation, 12h for price/volume/DDonchian.
Target: 50-150 total trades over 4 years (12-37/year). Focus on strong trends with volume confirmation to avoid choppy markets and reduce false breakouts.
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (Average Directional Index)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.zeros_like(close)
        minus_dm = np.zeros_like(close)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(close)
        plus_dm_smooth = np.zeros_like(close)
        minus_dm_smooth = np.zeros_like(close)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.zeros_like(close)
        minus_di = np.zeros_like(close)
        dx = np.zeros_like(close)
        
        for i in range(period, len(close)):
            if atr[i] > 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
                else:
                    dx[i] = 0
        
        # ADX: smoothed DX
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])  # First ADX value
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    midpoint = (highest_high + lowest_low) / 2.0
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, lookback)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(midpoint[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        high_ch = highest_high[i]
        low_ch = lowest_low[i]
        mid_pt = midpoint[i]
        
        # Trend regime: ADX > 25 = strong trend (good for breakout)
        is_strong_trend = adx_val > 25
        # Weak trend: ADX < 20 = avoid breakouts
        is_weak_trend = adx_val < 20
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and strong trend
            if price > high_ch and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike and strong trend
            elif price < low_ch and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR trend weakens
            if price <= mid_pt or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR trend weakens
            if price >= mid_pt or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_ADXTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
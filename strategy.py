#!/usr/bin/env python3
"""
Hypothesis: 1h 4h Donchian Breakout + Volume Spike + ADX Trend Filter.
Long when price breaks above 4h Donchian upper with volume > 1.8x average and ADX > 25.
Short when price breaks below 4h Donchian lower with volume > 1.8x average and ADX > 25.
Exit when price reverts to 4h Donchian midpoint or ADX < 20.
Uses 4h for Donchian channels and ADX, 1h for price/volume/timing.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels and ADX
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_4h, lower_4h = calculate_donchian(high_4h, low_4h, 20)
    midpoint_4h = (upper_4h + lower_4h) / 2.0
    
    # Calculate 4h ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            elif down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(high, np.nan)
        minus_di = np.full_like(high, np.nan)
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
        
        # DX and ADX
        dx = np.full_like(high, np.nan)
        for i in range(period, len(high)):
            if plus_di[i] + minus_di[i] > 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.full_like(high, np.nan)
        # Initial ADX
        adx[2*period-1] = np.mean(dx[period:2*period])
        # Wilder's smoothing for ADX
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Align 4h indicators to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available or outside session
        if (np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or 
            np.isnan(midpoint_4h_aligned[i]) or 
            np.isnan(adx_4h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_4h_aligned[i]
        upper = upper_4h_aligned[i]
        lower = lower_4h_aligned[i]
        midpoint = midpoint_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper with volume spike and strong trend
            if price > upper and vol_spike and adx_val > 25:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower with volume spike and strong trend
            elif price < lower and vol_spike and adx_val > 25:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR trend weakens (ADX < 20)
            if price <= midpoint or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to midpoint OR trend weakens (ADX < 20)
            if price >= midpoint or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_VolumeSpike_ADXTrend"
timeframe = "1h"
leverage = 1.0
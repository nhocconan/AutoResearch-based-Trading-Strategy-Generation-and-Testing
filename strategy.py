#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
Long when price breaks above 20-period Donchian high with volume > 1.3x average and ADX > 25.
Short when price breaks below 20-period Donchian low with volume > 1.3x average and ADX > 25.
Exit when price reverts to 10-period EMA or ADX < 20 (trend weakens).
Uses 4h for price/volume/Donchian/EMA, 1h for ADX to avoid look-ahead.
Target: 80-180 total trades over 4 years (20-45/year).
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
    
    # Get 1h data for ADX (to avoid look-ahead and use completed bars)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate 4h 10-period EMA for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1h ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        for i in range(1, len(high)):
            dm_plus[i] = max(high[i] - high[i-1], 0)
            dm_minus[i] = max(low[i-1] - low[i], 0)
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        tr_period = np.zeros_like(high)
        dm_plus_period = np.zeros_like(high)
        dm_minus_period = np.zeros_like(high)
        
        # Initial values
        tr_period[period] = np.mean(tr[1:period+1])
        dm_plus_period[period] = np.mean(dm_plus[1:period+1])
        dm_minus_period[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(high)):
            tr_period[i] = (tr_period[i-1] * (period-1) + tr[i]) / period
            dm_plus_period[i] = (dm_plus_period[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_period[i] = (dm_minus_period[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(high)
        di_minus = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if tr_period[i] != 0:
                di_plus[i] = 100 * dm_plus_period[i] / tr_period[i]
                di_minus[i] = 100 * dm_minus_period[i] / tr_period[i]
                if (di_plus[i] + di_minus[i]) != 0:
                    dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX: smoothed DX
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1h = calculate_adx(high_1h, low_1h, close_1h, 14)
    
    # Align 1h ADX to 4h timeframe
    adx_1h_aligned = align_htf_to_ltf(prices, df_1h, adx_1h)
    
    # Calculate volume spike (current volume > 1.3x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema_10[i]) or 
            np.isnan(adx_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1h_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema = ema_10[i]
        
        # Trend filter: ADX > 25 = strong trend, ADX < 20 = weak trend
        is_strong_trend = adx_val > 25
        is_weak_trend = adx_val < 20
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and strong trend
            if price > upper and vol_spike and is_strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume spike and strong trend
            elif price < lower and vol_spike and is_strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to EMA OR trend weakens
            if price <= ema or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to EMA OR trend weakens
            if price >= ema or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0
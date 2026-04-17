#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with volume confirmation and ADX trend filter.
Long when price breaks above upper Donchian channel with volume > 1.8x average and ADX > 25 (trending).
Short when price breaks below lower Donchian channel with volume > 1.8x average and ADX > 25.
Exit when price returns to the middle of the channel or ADX < 20 (range).
Uses 12h for price/volume/Donchian, 1d for ADX calculation to avoid whipsaw.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        middle = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
            middle[i] = (upper[i] + lower[i]) / 2.0
        
        return upper, lower, middle
    
    upper_12h, lower_12h, middle_12h = donchian_channels(high, low, 20)
    
    # Calculate volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(high)
        dm_minus = np.zeros_like(high)
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        atr = np.zeros_like(high)
        dm_plus_smooth = np.zeros_like(high)
        dm_minus_smooth = np.zeros_like(high)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        dm_plus_smooth[period] = np.mean(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.zeros_like(high)
        di_minus = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                di_plus[i] = (dm_plus_smooth[i] / atr[i]) * 100
                di_minus[i] = (dm_minus_smooth[i] / atr[i]) * 100
                if di_plus[i] + di_minus[i] > 0:
                    dx[i] = abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i]) * 100
                else:
                    dx[i] = 0
            else:
                di_plus[i] = 0
                di_minus[i] = 0
                dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h[i]) or 
            np.isnan(lower_12h[i]) or 
            np.isnan(middle_12h[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_1d_aligned[i]
        upper = upper_12h[i]
        lower = lower_12h[i]
        middle = middle_12h[i]
        
        # Trend regime: ADX > 25 = trending (good for breakout)
        is_trending = adx_val > 25
        # Range regime: ADX < 20 = ranging (avoid false breakouts)
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike in trending market
            if price > upper and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike in trending market
            elif price < lower and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle of channel OR market becomes ranging
            if price <= middle or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle of channel OR market becomes ranging
            if price >= middle or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ADXTrend"
timeframe = "12h"
leverage = 1.0
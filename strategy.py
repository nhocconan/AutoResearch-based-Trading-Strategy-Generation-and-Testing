#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and ADX trend filter
# Long when price breaks above Donchian(20) high with volume spike and ADX > 25
# Short when price breaks below Donchian(20) low with volume spike and ADX > 25
# Uses volume spike (current volume > 1.5 * 20-period average) to confirm breakouts
# ADX > 25 ensures we only trade in trending markets, reducing false breakouts
# Target: 20-50 trades per year to minimize fee drag while capturing strong moves
# Works in bull markets (breakouts continuation) and bear markets (breakdown continuations)

name = "4h_DonchianBreakout_VolumeSpike_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian(20) on 4h
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(20, len(high)):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
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
        plus_di = 100 * dm_plus_smooth / atr
        minus_di = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[np.isnan(dx) | np.isinf(dx)] = 0
        
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    volume_ma = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        volume_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with volume spike and ADX > 25
            if adx_1d_aligned[i] > 25 and volume_spike[i]:
                # Long breakout
                if close[i] > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown
                elif close[i] < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or ADX weakens
            if close[i] < lowest_low[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or ADX weakens
            if close[i] > highest_high[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
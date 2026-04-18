#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + Volume Spike + ADX Trend Filter
Hypothesis: Donchian breakouts capture momentum in trending markets, volume spikes confirm institutional participation, and ADX>25 filters for trending conditions. This combination works in both bull and bear markets by catching strong directional moves while avoiding choppy periods. Low trade frequency due to strict breakout conditions.
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
    
    # Get 1d data for higher timeframe context (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 4h
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    # Calculate ADX on 4h for trend strength filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values using Wilder's smoothing
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # Initial value
            result[period-1] = np.sum(data[:period])
            # Wilder smoothing
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / np.where(atr != 0, atr, 1)
        minus_di = 100 * wilder_smooth(minus_dm, period) / np.where(atr != 0, atr, 1)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        vol_ok = vol_spike[i]
        adx_ok = adx[i] > 25
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + volume spike + ADX > 25
            if close[i] > donch_upper[i] and vol_ok and adx_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower + volume spike + ADX > 25
            elif close[i] < donch_lower[i] and vol_ok and adx_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower or ADX weakens
            if close[i] < donch_lower[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper or ADX weakens
            if close[i] > donch_upper[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0
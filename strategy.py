#!/usr/bin/env python3
"""
Hypothesis: 1-hour ADX trend strength with 4-hour Donchian breakout and volume confirmation.
Long when ADX > 25 (trending) and price breaks above 4h Donchian upper band with volume spike.
Short when ADX > 25 and price breaks below 4h Donchian lower band with volume spike.
Exit when ADX < 20 (range) or price crosses 4h Donchian midline.
Designed for low trade frequency by requiring strong trend + breakout + volume confirmation.
Works in both bull and bear markets by following established trends.
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
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smooth_wmma(arr, period):
            result = np.full_like(arr, np.nan, dtype=float)
            if len(arr) < period:
                return result
            # First value is simple average
            result[period-1] = np.sum(arr[:period]) / period
            for i in range(period, len(arr)):
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        tr_14 = smooth_wmma(tr, period)
        plus_dm_14 = smooth_wmma(plus_dm, period)
        minus_dm_14 = smooth_wmma(minus_dm, period)
        
        # Directional Indicators
        plus_di_14 = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
        minus_di_14 = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di_14 + minus_di_14) != 0,
                      np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14) * 100, 0)
        adx = smooth_wmma(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Load 4h data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 20-period Donchian channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(low, np.nan, dtype=float)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donch_up_4h, donch_low_4h = donchian_channels(high_4h, low_4h, 20)
    donch_mid_4h = (donch_up_4h + donch_low_4h) / 2
    
    # Align Donchian channels to 1h
    donch_up_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_up_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    donch_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(donch_up_4h_aligned[i]) or 
            np.isnan(donch_low_4h_aligned[i]) or np.isnan(donch_mid_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: ADX > 25 (trending) and price breaks above 4h Donchian upper with volume spike
            if adx[i] > 25 and close[i] > donch_up_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: ADX > 25 and price breaks below 4h Donchian lower with volume spike
            elif adx[i] > 25 and close[i] < donch_low_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: ADX < 20 (range) or price crosses 4h Donchian midline
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 or price crosses below midline
                if adx[i] < 20 or close[i] < donch_mid_4h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 or price crosses above midline
                if adx[i] < 20 or close[i] > donch_mid_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_ADX_4H_Donchian_Breakout_Volume"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h Weekly Donchian Breakout with Volume Confirmation and ADX Filter
Hypothesis: Weekly trends are more stable and less prone to whipsaw. Combining weekly Donchian breakouts (price channel) with volume confirmation (institutional participation) and ADX > 20 (trending market) captures strong moves while avoiding choppy markets. Lower trade frequency due to 12h timeframe and strict conditions reduces fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    if len(tr) < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
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
    
    # Smoothed values
    def smooth_series(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = smooth_series(tr, period)
    plus_di = 100 * smooth_series(plus_dm, period) / np.where(atr != 0, atr, 1)
    minus_di = 100 * smooth_series(minus_dm, period) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = smooth_series(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate rolling max/min for Donchian channels
    donch_high = np.full_like(high_1w, np.nan)
    donch_low = np.full_like(low_1w, np.nan)
    
    for i in range(19, len(high_1w)):
        donch_high[i] = np.max(high_1w[i-19:i+1])
        donch_low[i] = np.min(low_1w[i-19:i+1])
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Calculate ADX on weekly data for trend strength
    adx_1w = calculate_adx(high_1w, low_1w, df_1w['close'].values, period=14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or np.isnan(adx_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        adx_val = adx_1w_aligned[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high + ADX > 20 + volume spike
            if (close[i] > upper and 
                adx_val > 20 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low + ADX > 20 + volume spike
            elif (close[i] < lower and 
                  adx_val > 20 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low or ADX weakens
            if close[i] < lower or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high or ADX weakens
            if close[i] > upper or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyDonchian_Breakout_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0
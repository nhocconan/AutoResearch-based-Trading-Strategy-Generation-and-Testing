#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout + Volume Spike + ADX Trend Filter
Hypothesis: Daily Donchian breakouts capture major trends. Volume spike confirms institutional participation.
ADX > 25 filters for trending markets, avoiding chop. Weekly trend filter ensures alignment with higher timeframe momentum.
Designed for low trade frequency (~15-25/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, period=20):
    """Calculate Donchian channels"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    for i in range(period - 1, len(high)):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.full_like(tr, np.nan)
    for i in range(period - 1, len(tr)):
        if i == period - 1:
            atr[i] = np.mean(tr[:period])
        else:
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate ADX"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period - 1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, period)
    plus_di = 100 * wilder_smooth(plus_dm, period) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilder_smooth(minus_dm, period) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilder_smooth(dx, period)
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX for trend filter
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_adx = calculate_adx(weekly_high, weekly_low, weekly_close, period=14)
    weekly_adx_aligned = align_htf_to_ltf(prices, df_weekly, weekly_adx)
    
    # Calculate daily Donchian channels
    donch_hi, donch_lo = calculate_donchian_channels(high, low, period=20)
    
    # Calculate daily ATR for stop loss
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume spike: current volume > 2.0x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(weekly_adx_aligned[i]) or np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        weekly_adx_val = weekly_adx_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + ADX > 25 + volume spike
            if (close[i] > donch_hi[i] and 
                weekly_adx_val > 25 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + ADX > 25 + volume spike
            elif (close[i] < donch_lo[i] and 
                  weekly_adx_val > 25 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: price crosses below Donchian low OR ATR-based stop
            if close[i] < donch_lo[i] or close[i] < (high[i] - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price crosses above Donchian high OR ATR-based stop
            if close[i] > donch_hi[i] or close[i] > (low[i] + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "1d"
leverage = 1.0
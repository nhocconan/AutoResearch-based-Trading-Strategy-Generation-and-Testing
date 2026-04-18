#!/usr/bin/env python3
"""
12h Donchian Breakout + Volume Spike + ADX Trend Filter
Hypothesis: Donchian breakouts capture strong trends in both bull and bear markets. Combined with volume spikes (institutional interest) and ADX > 20 (trending market), it filters out false breakouts. Weekly and daily timeframes provide higher timeframe trend confirmation. Low trade frequency (~15-30/year) minimizes fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian_channels(high, low, window):
    """Calculate Donchian Channels (upper, lower)"""
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(window-1, len(high)):
        upper[i] = np.max(high[i-window+1:i+1])
        lower[i] = np.min(low[i-window+1:i+1])
    
    return upper, lower

def calculate_adx(high, low, close, period):
    """Calculate ADX with Wilder smoothing"""
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
    
    # Wilder smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values use Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = WilderSmoothing(tr, period)
    plus_di = 100 * WilderSmoothing(plus_dm, period) / np.where(atr != 0, atr, 1)
    minus_di = 100 * WilderSmoothing(minus_dm, period) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = WilderSmoothing(dx, period)
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_upper, donch_lower = calculate_donchian_channels(high_1d, low_1d, 20)
    donch_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_lower)
    
    # Calculate ADX on 12h data
    adx_12h = calculate_adx(high, low, close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma = np.zeros_like(volume)
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
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or 
            np.isnan(adx_12h[i])):
            signals[i] = 0.0
            continue
        
        ema_50_val = ema_50_1w_aligned[i]
        donch_upper_val = donch_upper_aligned[i]
        donch_lower_val = donch_lower_aligned[i]
        adx_val = adx_12h[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + weekly uptrend + ADX > 20 + volume spike
            if (close[i] > donch_upper_val and 
                close[i] > ema_50_val and 
                adx_val > 20 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + weekly downtrend + ADX > 20 + volume spike
            elif (close[i] < donch_lower_val and 
                  close[i] < ema_50_val and 
                  adx_val > 20 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian or weekly trend turns down
            if close[i] < donch_lower_val or close[i] < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian or weekly trend turns up
            if close[i] > donch_upper_val or close[i] > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0
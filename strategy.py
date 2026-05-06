#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h ADX trend filter and volume confirmation
# Uses 4h ADX > 25 to identify trending markets (reduces whipsaw in ranges)
# 1h Donchian(20) breakout captures momentum in direction of 4h trend
# Volume > 1.5x 20-bar average confirms breakout strength
# Discrete sizing 0.20 to manage risk; target 60-150 total trades over 4 years (15-37/year)
# Works in both bull/bear: ADX filter ensures we only trade strong trends, breakouts capture momentum

name = "1h_Donchian20_4hADX25_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(14) trend filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed values
        tr_period = len(tr)
        atr = np.full_like(tr, np.nan)
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        
        # Wilder's smoothing (alpha = 1/period)
        if tr_period >= period:
            # Initial values
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_sum = np.nansum(plus_dm[1:period+1])
            minus_dm_sum = np.nansum(minus_dm[1:period+1])
            plus_di[period-1] = 100 * plus_dm_sum / atr[period-1] if atr[period-1] != 0 else 0
            minus_di[period-1] = 100 * minus_dm_sum / atr[period-1] if atr[period-1] != 0 else 0
            
            # Rolling smoothing
            for i in range(period, tr_period):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + plus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
                minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + minus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
        
        # Calculate DX and ADX
        dx = np.full_like(tr, np.nan)
        adx = np.full_like(tr, np.nan)
        
        for i in range(period, tr_period):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # Wilder's smoothing for ADX
        if tr_period >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, tr_period):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_filter = adx_4h > 25
    
    # Calculate 1h Donchian(20) channels
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Calculate volume confirmation (>1.5x 20-bar average)
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_filter)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper channel AND 4h ADX > 25 AND volume spike
            if close[i] > donchian_upper[i] and adx_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price < lower channel AND 4h ADX > 25 AND volume spike
            elif close[i] < donchian_lower[i] and adx_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < lower channel (reversal signal)
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > upper channel (reversal signal)
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
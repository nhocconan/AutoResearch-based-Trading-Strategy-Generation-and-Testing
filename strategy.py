#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter.
Long when price breaks above upper band with ADX>25 and volume spike.
Short when price breaks below lower band with ADX>25 and volume spike.
Exit when price crosses opposite Donchian band or ADX<20 (trend weakening).
Designed for 20-30 trades/year to minimize fee drag while capturing trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(len(high), np.nan)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index."""
    if len(high) < period * 2:
        return np.full(len(high), np.nan)
    
    # True Range
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    atr = np.full(len(high), np.nan)
    plus_di = np.full(len(high), np.nan)
    minus_di = np.full(len(high), np.nan)
    
    atr[period-1] = np.mean(tr[:period])
    plus_dm_sum = np.sum(plus_dm[:period])
    minus_dm_sum = np.sum(minus_dm[:period])
    
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_dm_smoothed = (plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]) if i == period else (plus_di[i-1] * (period - 1) + plus_dm[i]) / period
        minus_dm_smoothed = (minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]) if i == period else (minus_di[i-1] * (period - 1) + minus_dm[i]) / period
        
        if i == period:
            plus_dm_sum = plus_dm_smoothed * period
            minus_dm_sum = minus_dm_smoothed * period
        else:
            plus_dm_sum = plus_dm_smoothed * period
            minus_dm_sum = minus_dm_smoothed * period
        
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smoothed / atr[i]
            minus_di[i] = 100 * minus_dm_smoothed / atr[i]
        else:
            plus_di[i] = 0
            minus_di[i] = 0
    
    # DX and ADX
    dx = np.full(len(high), np.nan)
    adx = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, len(high)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def calculate_donchian(high, low, period):
    """Calculate Donchian Channels."""
    upper = np.full(len(high), np.nan)
    lower = np.full(len(high), np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align ADX to 4h timeframe
    adx_14_1d_4h = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate Donchian(20) on 4h
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need ADX and Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_1d_4h[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with ADX>25 and volume
            if close[i] > donchian_upper[i] and adx_14_1d_4h[i] > 25 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with ADX>25 and volume
            elif close[i] < donchian_lower[i] and adx_14_1d_4h[i] > 25 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower band or ADX weakens (<20)
            if close[i] < donchian_lower[i] or adx_14_1d_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band or ADX weakens (<20)
            if close[i] > donchian_upper[i] or adx_14_1d_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX14_Volume"
timeframe = "4h"
leverage = 1.0
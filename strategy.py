#!/usr/bin/env python3
"""
4H_RangeBreakout_With_Volume_Regime
Hypothesis: In ranging markets, price breaks of the 20-bar high/low with volume confirmation capture mean-reversion bounces; in trending markets, the same breaks capture continuation. A 1d ADX filter ensures we only trade when the daily trend is weak (ADX < 20), avoiding strong trends where breakouts fail. Works in bull markets by buying dips and selling rallies within ranges, and in bear markets by selling rallies and buying dips within ranges.
"""

name = "4H_RangeBreakout_With_Volume_Regime"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period high and low for breakout levels
    highest_20 = np.full_like(high, np.nan)
    lowest_20 = np.full_like(low, np.nan)
    for i in range(19, len(high)):
        highest_20[i] = np.max(high[i-19:i+1])
        lowest_20[i] = np.min(low[i-19:i+1])
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX on daily
    def calculate_dmi(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (EMA-like)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[:period] = np.mean(tr[:period])
        plus_dm_smooth[:period] = np.mean(plus_dm[:period])
        minus_dm_smooth[:period] = np.mean(minus_dm[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        # DX and ADX
        dx = np.zeros_like(high)
        dx[:2*period-1] = np.nan
        for i in range(2*period-1, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros_like(high)
        adx[:2*period-1] = np.nan
        for i in range(2*period, len(dx)):
            if not np.isnan(dx[i-1]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            else:
                adx[i] = np.nan
        
        return adx
    
    adx_14 = calculate_dmi(high_1d, low_1d, close_1d, 14)
    
    # Calculate volume average (20-period) for volume confirmation
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align 1d ADX to 4h timeframe
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Range condition: daily ADX < 20 (weak trend = ranging market)
        is_ranging = adx_14_aligned[i] < 20
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: price breaks above 20-period high in ranging market with volume
            if (close[i] > highest_20[i] and is_ranging and vol_confirm):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below 20-period low in ranging market with volume
            elif (close[i] < lowest_20[i] and is_ranging and vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below 20-period low or loses volume/range condition
            if (close[i] < lowest_20[i] or not is_ranging or not vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above 20-period high or loses volume/range condition
            if (close[i] > highest_20[i] or not is_ranging or not vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
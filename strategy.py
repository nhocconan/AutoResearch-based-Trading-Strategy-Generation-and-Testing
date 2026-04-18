#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with volume confirmation and 1d ADX trend filter.
Breakouts of the 20-period Donchian high/low capture momentum, while volume confirms institutional participation.
The ADX filter avoids false breakouts in ranging markets. Designed for 20-30 trades/year to minimize fee drag.
Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channel upper and lower bands."""
    upper = np.full_like(high, np.nan, dtype=float)
    lower = np.full_like(high, np.nan, dtype=float)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(high)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.maximum(high[1:] - low[1:], 
                   np.maximum(np.abs(high[1:] - close[:-1]), 
                              np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values
    atr = np.full(n, np.nan)
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    
    # Initial values
    if n >= period:
        atr[period-1] = np.nanmean(tr[1:period+1])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period+1])
        
        # Wilder smoothing
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Directional Indicators
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(period, n):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX
    adx = np.full(n, np.nan)
    if n >= 2*period-1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d data
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 4h timeframe
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    upper_20, lower_20 = calculate_donchian(high, low, 20)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(adx_1d_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX threshold
        trending = adx_1d_4h[i] >= 25
        
        if position == 0:
            # Only trade in trending markets
            if trending and vol_confirmed:
                # Long breakout above upper band
                if close[i] > upper_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below lower band
                elif close[i] < lower_20[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to middle of channel or opposite breakdown
            mid = (upper_20[i] + lower_20[i]) / 2
            if close[i] < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle of channel or opposite breakout
            mid = (upper_20[i] + lower_20[i]) / 2
            if close[i] > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX_Volume"
timeframe = "4h"
leverage = 1.0
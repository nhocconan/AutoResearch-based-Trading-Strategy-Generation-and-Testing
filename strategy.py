#!/usr/bin/env python3
"""
Hypothesis: 4h price breaks above/below 12h Donchian channel (20) with volume confirmation and 1d ADX trend filter.
In trending markets (ADX>25): breakouts continue, enter on high-volume breaks.
In ranging markets (ADX<20): fade extremes, enter on reversals from Donchian bands with volume.
Designed for 20-40 trades/year to minimize fee drag while capturing trends and reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM
    atr = np.full(len(high), np.nan)
    plus_dm_smooth = np.full(len(high), np.nan)
    minus_dm_smooth = np.full(len(high), np.nan)
    
    if len(high) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period])
        minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
    
    # Directional Indicators
    plus_di = np.full(len(high), np.nan)
    minus_di = np.full(len(high), np.nan)
    dx = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        if atr[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX
    adx = np.full(len(high), np.nan)
    if len(high) >= 2*period-1:
        adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_donchian(high, low, period=20):
    """Calculate Donchian channel upper and lower bands."""
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
    
    # Get 12h data for Donchian and ADX
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian(20)
    upper_12h, lower_12h = calculate_donchian(high_12h, low_12h, 20)
    
    # Calculate 12h ADX(14)
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align to 4h timeframe
    upper_12h_4h = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_4h = align_htf_to_ltf(prices, df_12h, lower_12h)
    adx_12h_4h = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_12h_4h[i]) or np.isnan(lower_12h_4h[i]) or 
            np.isnan(adx_12h_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Determine market regime
            if adx_12h_4h[i] > 25:  # Trending market
                # Long: break above upper Donchian with volume
                if close[i] > upper_12h_4h[i] and vol_confirmed:
                    signals[i] = 0.30
                    position = 1
                # Short: break below lower Donchian with volume
                elif close[i] < lower_12h_4h[i] and vol_confirmed:
                    signals[i] = -0.30
                    position = -1
            else:  # Ranging market (ADX <= 25)
                # Long: bounce from lower Donchian with volume
                if close[i] > lower_12h_4h[i] and close[i-1] <= lower_12h_4h[i-1] and vol_confirmed:
                    signals[i] = 0.30
                    position = 1
                # Short: rejection at upper Donchian with volume
                elif close[i] < upper_12h_4h[i] and close[i-1] >= upper_12h_4h[i-1] and vol_confirmed:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian or ADX drops (trend weakening)
            if close[i] < lower_12h_4h[i] or adx_12h_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian or ADX drops (trend weakening)
            if close[i] > upper_12h_4h[i] or adx_12h_4h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian12h_ADX_Volume"
timeframe = "4h"
leverage = 1.0
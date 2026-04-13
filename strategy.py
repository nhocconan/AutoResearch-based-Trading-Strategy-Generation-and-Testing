#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d trend filter (ADX>25) + volume confirmation
    # Long: price > Donchian(20) high + ADX>25 (trending) + volume > 2.0x 20-period average
    # Short: price < Donchian(20) low + ADX>25 (trending) + volume > 2.0x 20-period average
    # Exit: opposite Donchian breakout OR ADX<20 (range) OR Donchian middle cross
    # Uses 1d ADX for robust trend detection that works in both bull and bear markets
    # Volume spike confirms institutional interest in breakout direction
    # Target: 12-37 trades/year (50-150 over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) with min_periods
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period + 1:
            return np.full(n, np.nan)
        
        # True Range
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        tr = np.concatenate([[np.nan], tr])  # align indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        atr = np.full(n, np.nan)
        plus_dm_smooth = np.full(n, np.nan)
        minus_dm_smooth = np.full(n, np.nan)
        
        if n >= period:
            # Initial values
            atr[period-1] = np.nanmean(tr[1:period+1])
            plus_dm_smooth[period-1] = np.nanmean(plus_dm[1:period+1])
            minus_dm_smooth[period-1] = np.nanmean(minus_dm[1:period+1])
            
            # Wilder's smoothing
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
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX: smoothed DX
        adx = np.full(n, np.nan)
        if n >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, n):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 6h Donchian(20) for breakout with min_periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Get 6h volume for confirmation (>2.0x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Trend filter from 1d ADX
        trending = adx_1d_aligned[i] > 25
        ranging = adx_1d_aligned[i] < 20
        
        # Entry logic: Breakout + trending + volume confirmation
        long_entry = long_breakout and trending and volume_spike[i]
        short_entry = short_breakout and trending and volume_spike[i]
        
        # Exit logic: opposite breakout OR ranging market OR price crosses Donchian middle
        long_exit = short_breakout or ranging or (close[i] < donchian_mid[i])
        short_exit = long_breakout or ranging or (close[i] > donchian_mid[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_donchian_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0
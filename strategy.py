#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w ADX(14) trend filter.
In trending markets (ADX > 25): breakout above/below 20-day high/low with volume continuation.
In ranging markets (ADX < 20): fade extremes at Donchian bands with volume divergence.
Weekly ADX avoids choppy regimes, focusing on strong trends for breakout entries.
Designed for 10-20 trades/year to minimize fee drag on daily timeframe.
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
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = np.full(len(high), np.nan)
    dm_plus_smooth = np.full(len(high), np.nan)
    dm_minus_smooth = np.full(len(high), np.nan)
    
    if len(high) >= period:
        atr[period-1] = np.nanmean(tr[1:period])
        dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period])
        dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
    
    # DI and DX
    di_plus = np.full(len(high), np.nan)
    di_minus = np.full(len(high), np.nan)
    dx = np.full(len(high), np.nan)
    
    for i in range(period, len(high)):
        if atr[i] != 0:
            di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
            di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX
    adx = np.full(len(high), np.nan)
    for i in range(2*period-1, len(high)):
        if not np.isnan(dx[i-1]):
            if i == 2*period-1:
                adx[i] = np.nanmean(dx[period:i+1])
            else:
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate ADX(14) on 1w
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align to daily timeframe (since we're using 1d data primarily)
    donchian_high_1d = donchian_high  # already 1d
    donchian_low_1d = donchian_low    # already 1d
    adx_14_1w_1d = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    # Calculate volume moving average (20-period) on 1d
    vol_ma = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        vol_ma[i] = np.mean(close_1d[i-20:i])  # using close as proxy for volume MA
    
    # Align volume MA to 1d (it's already 1d)
    vol_ma_1d = vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need ADX and Donchian calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(adx_14_1w_1d[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        vol_confirmed = volume[i] > 1.2 * np.mean(volume[max(0, i-20):i])
        
        if position == 0:
            # Determine regime: trending (ADX > 25) or ranging (ADX < 20)
            if adx_14_1w_1d[i] > 25:
                # Trending regime: breakout entries
                if close[i] > donchian_high_1d[i] and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low_1d[i] and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
            elif adx_14_1w_1d[i] < 20:
                # Ranging regime: fade extremes
                if close[i] < donchian_low_1d[i] and vol_confirmed:
                    signals[i] = 0.25  # buy at support
                    position = 1
                elif close[i] > donchian_high_1d[i] and vol_confirmed:
                    signals[i] = -0.25  # sell at resistance
                    position = -1
        
        elif position == 1:
            # Long exit: opposite Donchian touch or ADX drops
            if close[i] < donchian_low_1d[i] or adx_14_1w_1d[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite Donchian touch or ADX drops
            if close[i] > donchian_high_1d[i] or adx_14_1w_1d[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wADX_Volume"
timeframe = "1d"
leverage = 1.0
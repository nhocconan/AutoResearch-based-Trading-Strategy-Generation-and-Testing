#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with volume confirmation and 1d ADX trend filter.
Trades breakouts of 20-period high/low only when accompanied by volume spike (>2x average)
and aligned with daily trend (ADX > 25). In ranging markets (ADX < 25), fades at channel
midpoint to avoid false breakouts. Designed for 20-30 trades/year to minimize fee drag.
Works in bull markets (buy breakouts, sell breakdowns) and bear markets (sell breakdowns,
buy bounces at support).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX indicator."""
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
    if n < 50:
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
    adx_1d_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channel (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need Donchian calculation and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_4h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        vol_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: ADX threshold
        trending = adx_1d_4h[i] >= 25
        ranging = adx_1d_4h[i] < 25
        
        if position == 0:
            # Trending market: breakout of Donchian channels
            if trending:
                # Long breakout above upper channel with volume
                if close[i] > donchian_high[i] and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below lower channel with volume
                elif close[i] < donchian_low[i] and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: fade at midpoint to avoid false breakouts
            else:
                midpoint = (donchian_high[i] + donchian_low[i]) / 2
                # Long near support (lower channel) with volume
                if close[i] <= donchian_low[i] * 1.001 and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                # Short near resistance (upper channel) with volume
                elif close[i] >= donchian_high[i] * 0.999 and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: reversal at midpoint or opposite channel
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reversal at midpoint or opposite channel
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0
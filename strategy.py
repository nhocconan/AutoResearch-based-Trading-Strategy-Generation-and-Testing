#!/usr/bin/env python3
"""
Hypothesis: 1d Weekly Range Breakout with Volume Confirmation and ADX Filter.
In trending markets (ADX >= 25 on weekly), price breaks out of the prior weekly range
with volume confirmation. In ranging markets (ADX < 25), mean reversion at weekly
support/resistance levels. Designed for 15-25 trades/year to minimize fee drag.
Works in bull markets (buy breakouts above weekly high) and bear markets (sell
breakdowns below weekly low).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX using Wilder's smoothing."""
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
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for range and ADX
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly ADX(14)
    adx_weekly = calculate_adx(high_weekly, low_weekly, close_weekly, 14)
    adx_weekly_daily = align_htf_to_ltf(prices, df_weekly, adx_weekly)
    
    # Calculate weekly range (prior week's high-low)
    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)
    for i in range(1, n):
        range_high[i] = high_weekly[i-1] if i-1 < len(high_weekly) else np.nan
        range_low[i] = low_weekly[i-1] if i-1 < len(low_weekly) else np.nan
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need weekly data and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_weekly_daily[i]) or np.isnan(range_high[i]) or 
            np.isnan(range_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX threshold
        trending = adx_weekly_daily[i] >= 25
        ranging = adx_weekly_daily[i] < 25
        
        if position == 0:
            if trending:
                # Trending market: breakout of weekly range
                if close[i] > range_high[i] and vol_confirmed:
                    signals[i] = 0.30
                    position = 1
                elif close[i] < range_low[i] and vol_confirmed:
                    signals[i] = -0.30
                    position = -1
            else:
                # Ranging market: mean reversion at weekly boundaries
                # Long near weekly low with volume
                if close[i] <= range_low[i] * 1.002 and vol_confirmed:
                    signals[i] = 0.30
                    position = 1
                # Short near weekly high with volume
                elif close[i] >= range_high[i] * 0.998 and vol_confirmed:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Long exit: return to weekly midpoint or opposite boundary
            midpoint = (range_high[i] + range_low[i]) / 2
            if close[i] >= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: return to weekly midpoint or opposite boundary
            midpoint = (range_high[i] + range_low[i]) / 2
            if close[i] <= midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_WeeklyRangeBreakout_Volume_ADX"
timeframe = "1d"
leverage = 1.0
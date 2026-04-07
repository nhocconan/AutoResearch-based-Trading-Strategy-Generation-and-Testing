#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Weekly Donchian Breakout with Volume and ADX Trend Filter
# Hypothesis: Breakouts of weekly Donchian channels (20-period) capture strong trends
# while avoiding false breakouts in ranging markets. Volume confirms breakout strength
# and ADX > 25 ensures trending conditions. Works in both bull and bear markets by
# taking breakouts in direction of higher timeframe trend.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "4h_weekly_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate upper and lower bands (20-period high/low)
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Align to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # ADX filter (14-period) on 4h data
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate +DM and -DM
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values (14-period)
    tr_period = 14
    atr = np.full(n, np.nan)
    dm_plus_smooth = np.full(n, np.nan)
    dm_minus_smooth = np.full(n, np.nan)
    
    # Initial values
    if n >= tr_period:
        atr[tr_period-1] = np.nansum(tr[:tr_period])
        dm_plus_smooth[tr_period-1] = np.nansum(dm_plus[:tr_period])
        dm_minus_smooth[tr_period-1] = np.nansum(dm_minus[:tr_period])
        
        # Wilder's smoothing
        for i in range(tr_period, n):
            atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(tr_period, n):
        if atr[i] != 0:
            plus_di[i] = (dm_plus_smooth[i] / atr[i]) * 100
            minus_di[i] = (dm_minus_smooth[i] / atr[i]) * 100
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
    
    # ADX (smoothed DX)
    adx = np.full(n, np.nan)
    adx_period = 14
    if n >= 2 * tr_period:
        adx[2*tr_period-1] = np.nansum(dx[tr_period:2*tr_period]) / tr_period
        for i in range(2*tr_period, n):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(250, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for trending market (ADX > 25)
        is_trending = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian band or ADX weakens
            if low[i] <= donchian_low_aligned[i] or not is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian band or ADX weakens
            if high[i] >= donchian_high_aligned[i] or not is_trending:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band with volume and trend
            if high[i] > donchian_high_aligned[i] and vol_filter[i] and is_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian band with volume and trend
            elif low[i] < donchian_low_aligned[i] and vol_filter[i] and is_trending:
                position = -1
                signals[i] = -0.25
    
    return signals
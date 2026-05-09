#!/usr/bin/env python3
# 12h_Weekly_Range_Breakout_Retest
# Hypothesis: Uses 1-week high/low as key support/resistance levels on 12h chart. Enters long when price
# breaks above weekly high with volume confirmation and retests the breakout level; enters short when
# price breaks below weekly low with volume confirmation and retests the breakdown level. Uses 1-day
# ADX to filter for trending markets (ADX > 25) to avoid false breakouts in ranging conditions.
# Designed to capture strong trending moves after consolidation periods. Target: 15-25 trades/year.

name = "12h_Weekly_Range_Breakout_Retest"
timeframe = "12h"
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
    
    # Get weekly data for range
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly high and low (using previous week's values)
    weekly_high = np.maximum.accumulate(high_1w)
    weekly_low = np.minimum.accumulate(low_1w)
    
    # Align weekly levels to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate 14-period ADX on daily timeframe
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
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
        atr = np.full_like(high, np.nan)
        dm_plus_smooth = np.full_like(high, np.nan)
        dm_minus_smooth = np.full_like(high, np.nan)
        
        if len(high) >= period:
            atr[period-1] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period+1])
            
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI+ and DI-
        di_plus = np.full_like(high, np.nan)
        di_minus = np.full_like(high, np.nan)
        valid_atr = (~np.isnan(atr)) & (atr != 0)
        di_plus[valid_atr] = (dm_plus_smooth[valid_atr] / atr[valid_atr]) * 100
        di_minus[valid_atr] = (dm_minus_smooth[valid_atr] / atr[valid_atr]) * 100
        
        # DX and ADX
        dx = np.full_like(high, np.nan)
        dx_valid = (~np.isnan(di_plus)) & (~np.isnan(di_minus)) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = (np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / 
                        (di_plus[dx_valid] + di_minus[dx_valid])) * 100
        
        adx = np.full_like(high, np.nan)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_level = np.full(n, np.nan)
    breakdown_level = np.full(n, np.nan)
    
    start_idx = max(30, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or \
           np.isnan(adx_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trending market filter: ADX > 25
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Check for breakout above weekly high
            if trending and close[i] > weekly_high_aligned[i] and volume_ratio[i] > 1.8:
                signals[i] = 0.25
                position = 1
                breakout_level[i] = weekly_high_aligned[i]
            # Check for breakdown below weekly low
            elif trending and close[i] < weekly_low_aligned[i] and volume_ratio[i] > 1.8:
                signals[i] = -0.25
                position = -1
                breakdown_level[i] = weekly_low_aligned[i]
        
        elif position == 1:
            # Trail stop: exit if price retests breakout level or breaks below weekly low
            if not np.isnan(breakout_level[i]) and close[i] <= breakout_level[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trail stop: exit if price retests breakdown level or breaks above weekly high
            if not np.isnan(breakdown_level[i]) and close[i] >= breakdown_level[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
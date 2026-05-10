#!/usr/bin/env python3
# 6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
# Hypothesis: Donchian channel breakouts on 6h timeframe with weekly pivot
# direction filter (from 1w timeframe) and volume confirmation capture
# strong trending moves while avoiding false breakouts. The weekly pivot
# provides a longer-term bias to filter trades, working in both bull
# (breakouts above resistance with bullish weekly bias) and bear
# (breakdowns below support with bearish weekly bias) markets.
# Volume spike ensures momentum behind the breakout.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for weekly pivot direction (longer-term bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d data for weekly pivot points (using prior week's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Weekly high, low, close from 1d data (need to aggregate to weekly)
    # Since we have 1d data, we can calculate weekly pivot using
    # the prior week's daily high, low, close
    # For simplicity, we'll use the prior week's daily values aggregated
    # But we need to be careful: we want the weekly pivot based on
    # the completed prior week
    
    # Instead, let's get weekly data directly if available
    # But per rules, we can use 1w from get_htf_data
    # So we'll use the 1w OHLC to calculate pivot for the current week
    # Actually, we want the pivot from the PRIOR week to avoid lookahead
    # So we'll use the 1w data shifted by 1 week
    
    # Weekly high, low, close from 1w data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot point (using prior week's data to avoid lookahead)
    # We'll shift the weekly data by 1 to use prior week's values
    # But we need to do this before alignment to avoid lookahead
    # So we calculate pivot using shifted weekly data
    
    # Shift weekly data by 1 to use prior week's OHLC
    weekly_high_prior = np.roll(weekly_high, 1)
    weekly_low_prior = np.roll(weekly_low, 1)
    weekly_close_prior = np.roll(weekly_close, 1)
    # Set first value to NaN since there's no prior week
    weekly_high_prior[0] = np.nan
    weekly_low_prior[0] = np.nan
    weekly_close_prior[0] = np.nan
    
    # Calculate pivot points using prior week's data
    weekly_pivot = (weekly_high_prior + weekly_low_prior + weekly_close_prior) / 3.0
    weekly_range = weekly_high_prior - weekly_low_prior
    R1 = weekly_pivot + (weekly_range)
    S1 = weekly_pivot - (weekly_range)
    R2 = weekly_pivot + 2 * weekly_range
    S2 = weekly_pivot - 2 * weekly_range
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Donchian channel (20-period) on 6h data
    # We need to calculate this ourselves since it's on the primary timeframe
    # Use pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for Donchian and weekly data
    
    for i in range(start_idx, n):
        # Check for NaN values in critical indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(R2_aligned[i]) or
            np.isnan(S2_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: break above Donchian high with bullish weekly bias (price > weekly pivot) and volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pivot_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with bearish weekly bias (price < weekly pivot) and volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below Donchian low or weekly pivot
            if (close[i] < donchian_low[i] or 
                close[i] < weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above Donchian high or weekly pivot
            if (close[i] > donchian_high[i] or 
                close[i] > weekly_pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
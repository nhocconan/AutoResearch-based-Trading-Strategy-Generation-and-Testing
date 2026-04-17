#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly pivot calculation (using previous week)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using previous week's data
    # We need to aggregate daily data into weeks
    n_days = len(high_1d)
    # Create week numbers (assuming data starts on Monday)
    week_nums = np.arange(n_days) // 7
    
    # Arrays to store weekly high, low, close
    weekly_high = np.full(n_days, np.nan)
    weekly_low = np.full(n_days, np.nan)
    weekly_close = np.full(n_days, np.nan)
    
    # Aggregate daily data into weekly candles
    for week in np.unique(week_nums[~np.isnan(week_nums)]):
        mask = (week_nums == week)
        if np.sum(mask) > 0:
            weekly_high[mask] = np.max(high_1d[mask])
            weekly_low[mask] = np.min(low_1d[mask])
            weekly_close[mask] = close_1d[mask][-1]  # Last day of week
    
    # Calculate weekly pivot points (using prior week's data to avoid look-ahead)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Shift to use previous week's pivots
    weekly_r1_prev = np.roll(weekly_r1, 1)
    weekly_s1_prev = np.roll(weekly_s1, 1)
    weekly_r2_prev = np.roll(weekly_r2, 1)
    weekly_s2_prev = np.roll(weekly_s2, 1)
    weekly_r1_prev[0] = np.nan
    weekly_s1_prev[0] = np.nan
    weekly_r2_prev[0] = np.nan
    weekly_s2_prev[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    weekly_r1_6h = align_htf_to_ltf(prices, df_1d, weekly_r1_prev)
    weekly_s1_6h = align_htf_to_ltf(prices, df_1d, weekly_s1_prev)
    weekly_r2_6h = align_htf_to_ltf(prices, df_1d, weekly_r2_prev)
    weekly_s2_6h = align_htf_to_ltf(prices, df_1d, weekly_s2_prev)
    
    # Volume confirmation: current volume > 2.0 * 12-period average (6h * 12 = 3 days)
    volume_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma12[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(weekly_r2_6h[i]) or 
            np.isnan(weekly_s2_6h[i]) or
            np.isnan(weekly_r1_6h[i]) or 
            np.isnan(weekly_s1_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 12-period average
        volume_filter = volume[i] > (2.0 * volume_ma12[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above weekly R2 with volume and volatility
            if close[i] > weekly_r2_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S2 with volume and volatility
            elif close[i] < weekly_s2_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below weekly R1 or volatility drops
            if close[i] < weekly_r1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above weekly S1 or volatility drops
            if close[i] > weekly_s1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R2_S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0
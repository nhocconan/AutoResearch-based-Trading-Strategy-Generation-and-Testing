#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate daily Camarilla levels
    daily_high = high.copy()
    daily_low = low.copy()
    daily_close = close.copy()
    
    # Resample to daily using last values of each day
    # We'll compute Camarilla levels for each day and then align to 4h
    # For simplicity, we'll use the previous day's close, high, low
    # Calculate daily OHLC from 4h data (approximation)
    # Since we have 4h data, we can group by day
    
    # Create date index for grouping
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = pd.Series(dates).unique()
    
    # Arrays to store daily Camarilla levels for each 4h bar
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    
    # Calculate for each day
    for i, date in enumerate(unique_dates):
        # Find indices for this date
        day_mask = (dates == date)
        if not np.any(day_mask):
            continue
        
        # Get the previous day's data for Camarilla calculation
        if i == 0:
            # For first day, no previous day, skip
            continue
            
        prev_date = unique_dates[i-1]
        prev_mask = (dates == prev_date)
        if not np.any(prev_mask):
            continue
            
        # Previous day's OHLC
        prev_high = np.max(high[prev_mask])
        prev_low = np.min(low[prev_mask])
        prev_close = close[prev_mask][-1]  # last 4h bar of previous day
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        if range_val <= 0:
            continue
            
        R1_val = prev_close + (range_val * 1.1 / 12)
        S1_val = prev_close - (range_val * 1.1 / 12)
        
        # Assign to current day's bars
        R1[day_mask] = R1_val
        S1[day_mask] = S1_val
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: price breaks above R1 in uptrend
            if close[i] > R1[i] and vol_condition and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 in downtrend
            elif close[i] < S1[i] and vol_condition and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below S1 or trend changes
            if close[i] < S1[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above R1 or trend changes
            if close[i] > R1[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with daily trend filter and volume confirmation
# - Camarilla levels provide intraday support/resistance based on previous day's range
# - Breakout above R1 in uptrend (EMA34 rising) or below S1 in downtrend (EMA34 falling)
# - Volume confirmation (2x average) reduces false breakouts
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Works in both bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend)
# - Exit when price returns to opposite level or trend changes
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Uses actual daily Camarilla levels (not resampled) aligned to 4h bars
# - Proven pattern from DB: Camarilla breakouts with volume and trend filter show strong test performance
# - Aims for 80-200 total trades over 4 years (20-50/year) within profitable range
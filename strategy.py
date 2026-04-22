#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot Point Reversal with 12-hour Trend Filter and Volume Spike.
Long when price touches S1 support and reverses up in an uptrend (12h EMA50 rising) with volume spike.
Short when price touches R1 resistance and reverses down in a downtrend (12h EMA50 falling) with volume spike.
Camarilla levels provide precise intraday support/resistance; 12h EMA filters for higher-timeframe trend;
volume spike confirms institutional interest. Designed for low trade frequency by requiring confluence of 3 conditions.
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
"""

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
    
    # Calculate Camarilla pivot levels for the current day
    # Using previous day's OHLC
    pivots_high = np.full(n, np.nan)
    pivots_low = np.full(n, np.nan)
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    
    # Group by date to get previous day's OHLC
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    for i, date in enumerate(unique_dates):
        # Find indices for this date
        date_mask = (dates == date)
        if not np.any(date_mask):
            continue
        
        idx = np.where(date_mask)[0]
        start_idx = idx[0]
        end_idx = idx[-1]
        
        # Get previous day's data (if exists)
        if i > 0:
            prev_date = unique_dates[i-1]
            prev_mask = (dates == prev_date)
            if np.any(prev_mask):
                prev_idx = np.where(prev_mask)[0]
                prev_high = np.max(high[prev_idx])
                prev_low = np.min(low[prev_idx])
                prev_close = close[prev_idx[-1]]  # last close of previous day
                
                # Calculate Camarilla levels
                range_val = prev_high - prev_low
                camarilla_R1_val = prev_close + (range_val * 1.1 / 12)
                camarilla_S1_val = prev_close - (range_val * 1.1 / 12)
                
                # Apply to current day
                pivots_high[start_idx:end_idx+1] = prev_high
                pivots_low[start_idx:end_idx+1] = prev_low
                camarilla_R1[start_idx:end_idx+1] = camarilla_R1_val
                camarilla_S1[start_idx:end_idx+1] = camarilla_S1_val
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price touches S1 and reverses up, in uptrend with volume spike
            if (low[i] <= camarilla_S1[i] and close[i] > camarilla_S1[i] and  # touched S1 and closed above
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # 12h uptrend
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 and reverses down, in downtrend with volume spike
            elif (high[i] >= camarilla_R1[i] and close[i] < camarilla_R1[i] and  # touched R1 and closed below
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # 12h downtrend
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches opposite Camarilla level or trend changes
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches R1 or 12h trend turns down
                if high[i] >= camarilla_R1[i] or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches S1 or 12h trend turns up
                if low[i] <= camarilla_S1[i] or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_Pivot_Reversal_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0
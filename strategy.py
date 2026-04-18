#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_12hEMA34_Volume_Conservative
Hypothesis: 4-hour breakouts above Camarilla R1 or below S1 with 12-hour EMA34 trend filter and volume confirmation. Conservative version with stricter volume and trend filters to reduce trade frequency and improve performance across BTC, ETH, and SOL in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-hour EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 with proper smoothing and min_periods
    ema34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema34_12h[i] = close_12h[i] * alpha + ema34_12h[i-1] * (1 - alpha)
    
    # Align 12-hour EMA34 to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Camarilla levels from previous day using proper daily aggregation
    # We'll use 12h data to approximate daily OHLC (2 periods = 1 day)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    # Calculate daily OHLC from 12h data (each day = 2 periods of 12h)
    # Create arrays for daily high, low, close
    daily_high = np.full(len(close_12h), np.nan)
    daily_low = np.full(len(close_12h), np.nan)
    daily_close = np.full(len(close_12h), np.nan)
    
    # For each 12h period, we need the daily context
    # Since 12h data: index 0,1 = day 0; index 2,3 = day 1; etc.
    for i in range(len(close_12h)):
        day_idx = i // 2  # Each day has 2 periods of 12h data
        start_idx = day_idx * 2
        end_idx = start_idx + 2
        
        # Get the two 12h periods for this day
        if end_idx <= len(close_12h):
            day_high = np.max(df_12h['high'].values[start_idx:end_idx])
            day_low = np.min(df_12h['low'].values[start_idx:end_idx])
            day_close = df_12h['close'].values[end_idx-1]  # Close of last period
            
            daily_high[i] = day_high
            daily_low[i] = day_low
            daily_close[i] = day_close
    
    # Now calculate Camarilla levels for each 4h bar using the prior day's OHLC
    for i in range(n):
        # Find which 12h period corresponds to this 4h bar
        # 4h to 12h ratio: 3 periods of 4h = 1 period of 12h
        period_12h_idx = i // 3
        
        # We need the previous day's data (not current day)
        prev_day_idx = period_12h_idx - 2  # Go back 2 periods (1 day) in 12h data
        
        if prev_day_idx >= 0 and not (np.isnan(daily_high[prev_day_idx]) or np.isnan(daily_low[prev_day_idx])):
            prev_high = daily_high[prev_day_idx]
            prev_low = daily_low[prev_day_idx]
            prev_close = daily_close[prev_day_idx]
            range_val = prev_high - prev_low
            
            camarilla_r1[i] = prev_close + range_val * 1.1 / 12
            camarilla_s1[i] = prev_close - range_val * 1.1 / 12
    
    # Volume spike: current volume > 2.0 x 30-period average (stricter)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30, 2)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and 12h uptrend
            if (close[i] > camarilla_r1[i] and vol_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and 12h downtrend
            elif (close[i] < camarilla_s1[i] and vol_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Camarilla S1 or 12h trend turns down
            if (close[i] < camarilla_s1[i] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Camarilla R1 or 12h trend turns up
            if (close[i] > camarilla_r1[i] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_12hEMA34_Volume_Conservative"
timeframe = "4h"
leverage = 1.0
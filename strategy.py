#!/usr/bin/env python3
"""
1d_WeeklyPivot_R1S1_Breakout_Volume_RangeFilter
Hypothesis: On 1d timeframe, use weekly Camarilla pivot levels (R1, S1) for breakout signals, filtered by weekly RSI and daily volume confirmation. Long when price breaks above R1 with volume > 1.5x 20-day average and weekly RSI > 50; short when price breaks below S1 with volume > 1.5x 20-day average and weekly RSI < 50. Exit on opposite break. This captures momentum while avoiding false breakouts in ranging markets. Targets 10-20 trades/year by requiring multiple confluence factors, with position size 0.25.
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
    
    # Calculate 20-day average volume for filtering
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    
    # Get weekly data for Camarilla pivots and RSI
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate Camarilla pivot levels for each week
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(len(weekly_close), np.nan)
    camarilla_s1 = np.full(len(weekly_close), np.nan)
    
    for i in range(len(weekly_close)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            rang = weekly_high[i] - weekly_low[i]
            camarilla_r1[i] = weekly_close[i] + rang * 1.1 / 12
            camarilla_s1[i] = weekly_close[i] - rang * 1.1 / 12
    
    # Calculate weekly RSI(14)
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(weekly_close, np.nan)
    avg_loss = np.full_like(weekly_close, np.nan)
    
    for i in range(14, len(weekly_close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:14])
            avg_loss[i] = np.mean(loss[0:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    weekly_rsi = 100 - (100 / (1 + rs))
    
    # Align weekly data to daily timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s1)
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma20[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(weekly_rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and RSI > 50
            if (close[i] > camarilla_r1_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                weekly_rsi_aligned[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation and RSI < 50
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  weekly_rsi_aligned[i] < 50):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below S1
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_R1S1_Breakout_Volume_RangeFilter"
timeframe = "1d"
leverage = 1.0
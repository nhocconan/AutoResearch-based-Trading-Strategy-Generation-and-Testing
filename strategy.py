#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_1wTrend_Volume_Strict
Hypothesis: On 12h timeframe, buy when price breaks above Camarilla R3 level with weekly uptrend (weekly close > weekly EMA34) and volume confirmation; sell when breaks below S3 level with weekly downtrend (weekly close < weekly EMA34) and volume confirmation. Uses weekly EMA34 for trend filter to avoid whipsaws and volume spike for confirmation. Designed for 12h timeframe with expected 50-150 trades over 4 years to minimize fee drag while capturing trends in both bull and bear markets.
"""
name = "12h_Camarilla_R3S3_1wTrend_Volume_Strict"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Calculate Camarilla levels from previous week's OHLC
    # Camarilla levels use previous week's data
    prev_close = df_weekly['close'].shift(1).values
    prev_high = df_weekly['high'].shift(1).values
    prev_low = df_weekly['low'].shift(1).values
    
    # Calculate Camarilla levels for each week
    # R3 = C + (H-L)*1.1/2
    # S3 = C - (H-L)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s3)
    
    # Volume filter: current volume > 1.5 * 50-period average volume
    vol_avg = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema34_weekly_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + weekly uptrend + volume filter
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_weekly_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + weekly downtrend + volume filter
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_weekly_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level
            if position == 1:
                if close[i] < camarilla_s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeFilter
Hypothesis: Use 1w Camarilla R3/S3 levels as breakout levels on 1d timeframe, filtered by 1w EMA trend and volume spike.
Long when price breaks above weekly R3 and close > weekly EMA34 with volume > 2x average.
Short when price breaks below weekly S3 and close < weekly EMA34 with volume > 2x average.
Camarilla levels provide institutional support/resistance, weekly EMA filters trend, volume confirms breakout strength.
Designed for 1d timeframe to target 15-25 trades/year with low frequency to minimize fee drag in bear markets.
"""
name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (R3, S3)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    weekly_range = df_1w['high'] - df_1w['low']
    camarilla_r3 = df_1w['close'] + 1.1 * weekly_range
    camarilla_s3 = df_1w['close'] - 1.1 * weekly_range
    camarilla_r3_vals = camarilla_r3.values
    camarilla_s3_vals = camarilla_s3.values
    
    # Calculate weekly EMA34 for trend filter
    weekly_ema34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly indicators to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_vals)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_vals)
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema34)
    
    # Volume filter: current volume > 2x 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # days since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(weekly_ema34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 10 days between trades to reduce frequency
            if bars_since_exit < 10:
                continue
                
            # Long: price breaks above weekly R3 + close > weekly EMA34 + volume filter
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > weekly_ema34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below weekly S3 + close < weekly EMA34 + volume filter
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < weekly_ema34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1 and (close[i] < camarilla_s3_aligned[i] or close[i] < weekly_ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (close[i] > camarilla_r3_aligned[i] or close[i] > weekly_ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
#!/usr/bin/env python3
name = "1d_1w_Camarilla_R3_S3_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Camarilla levels (R3, S3)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous week's close
    prev_week_close = df_1w['close'].shift(1).values  # Previous week close
    prev_week_high = df_1w['high'].shift(1).values    # Previous week high
    prev_week_low = df_1w['low'].shift(1).values      # Previous week low
    
    # Camarilla R3 and S3 levels
    R3 = prev_week_close + (prev_week_high - prev_week_low) * 1.1 / 2
    S3 = prev_week_close - (prev_week_high - prev_week_low) * 1.1 / 2
    
    # Align weekly levels to daily
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Volume filter: current volume > 1.3x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 5  # ~5 days to prevent overtrading
    
    start_idx = max(20, 1)  # Need 20 days for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above weekly R3 with volume confirmation
            if close[i] > R3_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below weekly S3 with volume confirmation
            elif close[i] < S3_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below weekly S3
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above weekly R3
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On daily timeframe, price breaking above/below weekly Camarilla R3/S3 levels with volume confirmation captures institutional breakout moves. Weekly Camarilla levels act as significant support/resistance that institutions watch. Volume filter ensures breakouts have conviction. This strategy works in both bull markets (breakouts above R3 in uptrend) and bear markets (breakdowns below S3 in downtrend) by capturing meaningful price levels. Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag while capturing significant moves. Uses 1d timeframe with 1h Camarilla levels.
#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Trade 4-hour breakouts of Camarilla R3/S3 levels only when aligned with 12-hour trend (EMA50) and confirmed by volume spike (>2x average). Uses 12h timeframe for trend direction and 4h for precise entry. Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation. Targets 20-40 trades/year with low fee impact.
"""

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeS"
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
    
    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    daily_high = df_12h['high'].values
    daily_low = df_12h['low'].values
    daily_close = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h period's range
    prev_daily_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_daily_low = np.concatenate([[np.nan], daily_low[:-1]])
    prev_daily_close = np.concatenate([[np.nan], daily_close[:-1]])
    
    # Calculate Camarilla levels
    R3 = prev_daily_close + 1.1 * (prev_daily_high - prev_daily_low) / 2
    S3 = prev_daily_close - 1.1 * (prev_daily_high - prev_daily_low) / 2
    R4 = prev_daily_close + 1.1 * (prev_daily_high - prev_daily_low)
    S4 = prev_daily_close - 1.1 * (prev_daily_high - prev_daily_low)
    
    # Align Camarilla levels to 4h timeframe (need to wait for 12h close)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    
    # Get 4h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or 
            np.isnan(S4_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_12h, daily_close)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = daily_close_aligned[i] > ema_50_12h_aligned[i]
        trend_down = daily_close_aligned[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R3 with upward trend and volume spike
            if (close[i] > R3_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S3 with downward trend and volume spike
            elif (close[i] < S3_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S3 level or trend turns down
            if close[i] < S3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R3 level or trend turns up
            if close[i] > R3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_1dTrend_VolumeSpike_v1
Hypothesis: Trade 6-hour breakouts of Camarilla R3/S3 levels only when aligned with daily trend (EMA34) and confirmed by volume spike (>2x average). Uses daily timeframe for trend direction and 6h for precise entry. Designed to work in both bull and bear markets by requiring trend alignment and volume confirmation. Targets 15-35 trades/year with low fee impact.
"""

name = "6h_Camarilla_R3S3_1dTrend_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Get daily data for trend filter and Camarilla calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 34:
        return np.zeros(n)
    
    daily_high = df_d['high'].values
    daily_low = df_d['low'].values
    daily_close = df_d['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34_d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_d_aligned = align_htf_to_ltf(prices, df_d, ema_34_d)
    
    # Calculate Camarilla levels from previous day's range
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    prev_daily_high = np.concatenate([[np.nan], daily_high[:-1]])
    prev_daily_low = np.concatenate([[np.nan], daily_low[:-1]])
    prev_daily_close = np.concatenate([[np.nan], daily_close[:-1]])
    
    # Calculate Camarilla levels
    R3 = prev_daily_close + 1.1 * (prev_daily_high - prev_daily_low) / 2
    S3 = prev_daily_close - 1.1 * (prev_daily_high - prev_daily_low) / 2
    R4 = prev_daily_close + 1.1 * (prev_daily_high - prev_daily_low)
    S4 = prev_daily_close - 1.1 * (prev_daily_high - prev_daily_low)
    
    # Align Camarilla levels to 6h timeframe (need to wait for daily close)
    R3_aligned = align_htf_to_ltf(prices, df_d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_d, S4)
    
    # Get 6h volume for confirmation
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
            np.isnan(ema_34_d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_d, daily_close)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = daily_close_aligned[i] > ema_34_d_aligned[i]
        trend_down = daily_close_aligned[i] < ema_34_d_aligned[i]
        
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
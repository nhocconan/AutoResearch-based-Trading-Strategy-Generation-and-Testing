#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v3
Hypothesis: Trade breakouts at Camarilla R3/S3 levels on 12h chart with daily EMA34 trend filter and volume confirmation.
Breakouts capture strong directional moves while aligning with higher timeframe trend to avoid counter-trend trades.
Volume spike (>2x 20-period average) confirms institutional participation. Designed for low trade frequency (12-37/year) to minimize fee drift.
Works in both bull and bear markets by trading only in direction of daily trend.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume_v3"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly data for additional trend confirmation (optional)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels for 12h: based on previous bar's range
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous bar's OHLC to avoid look-ahead
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_width = 1.1 * (prev_high - prev_low) / 2
    r3_level = prev_close + camarilla_width
    s3_level = prev_close - camarilla_width
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA and Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_level[i]) or np.isnan(s3_level[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get daily close for trend determination
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_34_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_34_1d_aligned[i]
        
        # Weekly trend filter (optional, can be removed if too restrictive)
        weekly_trend_up = True
        weekly_trend_down = True
        if len(df_1w) > 0:
            weekly_close_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
            if not np.isnan(weekly_close_aligned[i]) and not np.isnan(ema_34_1w_aligned[i]):
                weekly_trend_up = weekly_close_aligned[i] > ema_34_1w_aligned[i]
                weekly_trend_down = weekly_close_aligned[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, volume spike, daily trend up, weekly trend up
            if (close[i] > r3_level[i] and 
                vol_ratio[i] > 2.0 and 
                daily_trend_up and 
                weekly_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, volume spike, daily trend down, weekly trend down
            elif (close[i] < s3_level[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down and 
                  weekly_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 or trend changes
            if close[i] < s3_level[i] or not (daily_trend_up and weekly_trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 or trend changes
            if close[i] > r3_level[i] or not (daily_trend_down and weekly_trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
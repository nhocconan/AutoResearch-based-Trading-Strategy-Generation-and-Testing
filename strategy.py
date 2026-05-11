#!/usr/bin/env python3
name = "6h_WeeklyPivot_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop (weekly pivot requires daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from daily data
    # Weekly high/low/close from daily data (simplified: use last 5 days)
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
    
    # Pivot point = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Support and resistance levels
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align to 6h timeframe (wait for weekly close - use 2-bar delay for weekly confirmation)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot, additional_delay_bars=2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=2)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2, additional_delay_bars=2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2, additional_delay_bars=2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=2)
    
    # Trend filter: 50-period EMA on 1d
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume surge filter (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long entry: break above R3 in uptrend with volume
            if (uptrend and 
                close[i] > r3_aligned[i] and 
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short entry: break below S3 in downtrend with volume
            elif (downtrend and 
                  close[i] < s3_aligned[i] and 
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot level
            if position == 1:
                if close[i] <= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] >= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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
    
    # 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla pivot levels from 1d data
    # Formula: P = (H+L+C)/3, range = H-L
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    daily_range = daily_high - daily_low
    r3 = daily_close + daily_range * 1.1 / 2
    s3 = daily_close - daily_range * 1.1 / 2
    r4 = daily_close + daily_range * 1.1
    s4 = daily_close - daily_range * 1.1
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily trend: price relative to pivot (using previous day's pivot)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    daily_trend_up = close > pivot_aligned
    daily_trend_down = close < pivot_aligned
    
    # Volume filter: volume > 1.5x 20-period average on 6h
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours to prevent overtrading
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = daily_trend_up[i]
        trend_down = daily_trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R3 with volume in daily uptrend
            if (close[i] > r3_aligned[i] and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S3 with volume in daily downtrend
            elif (close[i] < s3_aligned[i] and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below R3 or daily trend changes to down
            if close[i] < r3_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above S3 or daily trend changes to up
            if close[i] > s3_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 6h timeframe, price breaking above/below Camarilla R3/S3 levels with volume confirmation and daily trend filter captures institutional breakout momentum. R3/S3 represent strong intraday support/resistance where breakouts often initiate significant moves. Daily trend filter ensures we trade in the direction of the higher timeframe trend, reducing false signals. Works in bull markets (breakouts above R3 in daily uptrend) and bear markets (breakdowns below S3 in daily downtrend). Volume spike confirms institutional participation. Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while capturing significant moves. Camarilla levels are widely watched by institutional traders, providing edge in liquid markets like BTC/ETH.
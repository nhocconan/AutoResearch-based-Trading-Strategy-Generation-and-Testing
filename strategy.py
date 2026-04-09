#!/usr/bin/env python3
# 6h_1d_weekly_pivot_breakout_v4
# Hypothesis: 6-hour breakout of weekly pivot levels with daily EMA200 trend filter and volume confirmation.
# Long when price breaks above R3 resistance with price > daily EMA200 and volume > 2.5x 20-bar average.
# Short when price breaks below S3 support with price < daily EMA200 and volume > 2.5x 20-bar average.
# Exit when price returns to opposite pivot level (S3 for longs, R3 for shorts).
# Weekly pivots calculated from prior week's OHLC: PP=(H+L+C)/3, R3=H+2*(PP-L), S3=L-2*(H-PP).
# Works in bull markets via breakout continuation and in bear markets via mean reversion at extreme levels.
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_weekly_pivot_breakout_v4"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    pp = np.full(len(df_w), np.nan)
    r3 = np.full(len(df_w), np.nan)
    s3 = np.full(len(df_w), np.nan)
    for i in range(1, len(df_w)):  # Start from 1 to use previous week
        ph = df_w['high'].iloc[i-1]
        pl = df_w['low'].iloc[i-1]
        pc = df_w['close'].iloc[i-1]
        pp[i] = (ph + pl + pc) / 3.0
        r3[i] = ph + 2 * (pp[i] - pl)
        s3[i] = pl - 2 * (ph - pp[i])
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    
    # Load daily data ONCE before loop
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA200 for trend filter
    close_d = df_d['close'].values
    ema_200_d = np.full(len(close_d), np.nan)
    if len(close_d) >= 200:
        # Initialize SMA for first 200 periods
        sma = np.mean(close_d[:200])
        ema_200_d[199] = sma
        multiplier = 2 / (200 + 1)
        for i in range(200, len(close_d)):
            ema = (close_d[i] - ema_200_d[i-1]) * multiplier + ema_200_d[i-1]
            ema_200_d[i] = ema
    
    # Align daily EMA200 to 6h timeframe
    ema_200_d_aligned = align_htf_to_ltf(prices, df_d, ema_200_d)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_200_d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below S3 level
            if close[i] <= s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above R3 level
            if close[i] >= r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above R3 with trend and volume filters
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_200_d_aligned[i] and 
                volume[i] > vol_ma_20[i] * 2.5):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below S3 with trend and volume filters
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_200_d_aligned[i] and 
                  volume[i] > vol_ma_20[i] * 2.5):
                position = -1
                signals[i] = -0.25
    
    return signals
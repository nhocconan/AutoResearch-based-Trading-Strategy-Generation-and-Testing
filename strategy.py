#!/usr/bin/env python3
# 6h_1w_1d_pivots_breakout_volume_v2
# Hypothesis: Use weekly and daily pivot levels with volume confirmation on 6h timeframe.
# Long when price breaks above weekly R3 or daily R3 with volume surge and weekly trend up.
# Short when price breaks below weekly S3 or daily S3 with volume surge and weekly trend down.
# Pivots act as institutional support/resistance; volume confirms breakout strength.
# Weekly trend filter prevents counter-trend trades. Target: 50-150 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_pivots_breakout_volume_v2"
timeframe = "6h"
leverage = 1.0

def calculate_pivots(high, low, close):
    """Calculate standard pivot points and support/resistance levels."""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivots from previous week
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # We need previous week's data to calculate pivots for current week
    wk_pivot, wk_r1, wk_r2, wk_r3, wk_s1, wk_s2, wk_s3 = calculate_pivots(
        wk_high[:-1], wk_low[:-1], wk_close[:-1]
    )
    # Prepend first value to maintain same length
    wk_pivot = np.concatenate([[np.nan], wk_pivot])
    wk_r1 = np.concatenate([[np.nan], wk_r1])
    wk_r2 = np.concatenate([[np.nan], wk_r2])
    wk_r3 = np.concatenate([[np.nan], wk_r3])
    wk_s1 = np.concatenate([[np.nan], wk_s1])
    wk_s2 = np.concatenate([[np.nan], wk_s2])
    wk_s3 = np.concatenate([[np.nan], wk_s3])
    
    # Align weekly pivot levels to 6h timeframe
    wk_pivot_aligned = align_htf_to_ltf(prices, df_1w, wk_pivot)
    wk_r3_aligned = align_htf_to_ltf(prices, df_1w, wk_r3)
    wk_s3_aligned = align_htf_to_ltf(prices, df_1w, wk_s3)
    
    # Weekly trend: EMA25/50 crossover on weekly close
    wk_ema25 = pd.Series(wk_close).ewm(span=25, adjust=False, min_periods=25).mean().values
    wk_ema50 = pd.Series(wk_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Shift to use only previous week's data for trend
    wk_ema25 = np.concatenate([[np.nan], wk_ema25[:-1]])
    wk_ema50 = np.concatenate([[np.nan], wk_ema50[:-1]])
    wk_ema25_aligned = align_htf_to_ltf(prices, df_1w, wk_ema25)
    wk_ema50_aligned = align_htf_to_ltf(prices, df_1w, wk_ema50)
    
    # Get daily data for additional pivot confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivots from previous day
    dy_high = df_1d['high'].values
    dy_low = df_1d['low'].values
    dy_close = df_1d['close'].values
    
    dy_pivot, dy_r1, dy_r2, dy_r3, dy_s1, dy_s2, dy_s3 = calculate_pivots(
        dy_high[:-1], dy_low[:-1], dy_close[:-1]
    )
    # Prepend first value to maintain same length
    dy_pivot = np.concatenate([[np.nan], dy_pivot])
    dy_r1 = np.concatenate([[np.nan], dy_r1])
    dy_r2 = np.concatenate([[np.nan], dy_r2])
    dy_r3 = np.concatenate([[np.nan], dy_r3])
    dy_s1 = np.concatenate([[np.nan], dy_s1])
    dy_s2 = np.concatenate([[np.nan], dy_s2])
    dy_s3 = np.concatenate([[np.nan], dy_s3])
    
    # Align daily pivot levels to 6h timeframe
    dy_r3_aligned = align_htf_to_ltf(prices, df_1d, dy_r3)
    dy_s3_aligned = align_htf_to_ltf(prices, df_1d, dy_s3)
    
    # Volume confirmation: 6h volume > 2.0x 24-period average (4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(wk_r3_aligned[i]) or np.isnan(wk_s3_aligned[i]) or
            np.isnan(dy_r3_aligned[i]) or np.isnan(dy_s3_aligned[i]) or
            np.isnan(wk_ema25_aligned[i]) or np.isnan(wk_ema50_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_24[i] if vol_ma_24[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price breaks below weekly S3 or daily S3
            if close[i] < wk_s3_aligned[i] or close[i] < dy_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above weekly R3 or daily R3
            if close[i] > wk_r3_aligned[i] or close[i] > dy_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above weekly R3 or daily R3 with volume surge and weekly uptrend
            weekly_uptrend = wk_ema25_aligned[i] > wk_ema50_aligned[i]
            long_breakout = (close[i] > wk_r3_aligned[i] or close[i] > dy_r3_aligned[i])
            if long_breakout and vol_surge and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: Break below weekly S3 or daily S3 with volume surge and weekly downtrend
            elif (close[i] < wk_s3_aligned[i] or close[i] < dy_s3_aligned[i]) and \
                 vol_surge and (wk_ema25_aligned[i] < wk_ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
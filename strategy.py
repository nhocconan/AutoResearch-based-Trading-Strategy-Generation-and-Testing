#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly pivot levels from 1w timeframe with
# 60-period EMA trend filter and volume confirmation. Uses price rejection at
# weekly R2/S2 levels in ranging markets and breakouts beyond R3/S3 in trending
# markets. Designed to work in both bull and bear by adapting to weekly structure
# and avoiding choppy periods via volume filters. Target: 15-25 trades/year.

name = "6h_WeeklyPivot_R2S2_Rejection_R3S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels (primary filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot and levels from previous week's OHLC
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Shift by 1 to use previous week's data
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    # Set first value to NaN since no previous week exists
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    prev_weekly_range = prev_high_1w - prev_low_1w
    pivot = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r1 = pivot + 1.1 * prev_weekly_range / 6
    r2 = pivot + 1.1 * prev_weekly_range / 4
    r3 = pivot + 1.1 * prev_weekly_range / 2
    s1 = pivot - 1.1 * prev_weekly_range / 6
    s2 = pivot - 1.1 * prev_weekly_range / 4
    s3 = pivot - 1.1 * prev_weekly_range / 2
    
    # Align weekly levels to 6h
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 60-period EMA for trend filter (using daily data)
    ema60_1d = pd.Series(df_1d['close']).ewm(span=60, adjust=False, min_periods=60).mean().values
    ema60_6h = align_htf_to_ltf(prices, df_1d, ema60_1d)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(ema60_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long conditions:
            # 1. Rejection at S2 (price < S2 and closes back above S2) in ranging market
            # 2. Breakout above R3 in trending market (price > EMA60)
            rejection_long = (low[i] < s2_6h[i] and close[i] > s2_6h[i])
            breakout_long = (close[i] > r3_6h[i] and close[i] > ema60_6h[i])
            
            if (rejection_long or breakout_long) and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # Short conditions:
            # 1. Rejection at R2 (price > R2 and closes back below R2) in ranging market
            # 2. Breakout below S3 in trending market (price < EMA60)
            rejection_short = (high[i] > r2_6h[i] and close[i] < r2_6h[i])
            breakout_short = (close[i] < s3_6h[i] and close[i] < ema60_6h[i])
            
            if (rejection_short or breakout_short) and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price fails to hold above S2 OR trend turns down
            if close[i] < s2_6h[i] or close[i] < ema60_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price fails to hold below R2 OR trend turns up
            if close[i] > r2_6h[i] or close[i] > ema60_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_HTF_Pivot_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for weekly pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly pivot using daily data (approximated via weekly high/low/close)
    # Calculate weekly high/low/close from daily data (simplified)
    # We'll use 5-day rolling window to approximate weekly
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    weekly_high = pd.Series(daily_high).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(daily_low).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(daily_close).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot point and support/resistance levels
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    r3 = weekly_pivot + (weekly_range * 1.1)  # R3 level
    s3 = weekly_pivot - (weekly_range * 1.1)  # S3 level
    
    # Align weekly levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Weekly trend filter: price above/below weekly pivot
    weekly_trend = (weekly_close > weekly_pivot).astype(float)
    weekly_trend_6h = align_htf_to_ltf(prices, df_1d, weekly_trend)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    # Price distance filter: require breakout to be at least 0.3% above/below level
    price_above_r3 = close > r3_6h * 1.003
    price_below_s3 = close < s3_6h * 0.997
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA and weekly calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(weekly_trend_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and weekly uptrend
            long_cond = (price_above_r3[i] and vol_spike[i] and weekly_trend_6h[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and weekly downtrend
            short_cond = (price_below_s3[i] and vol_spike[i] and weekly_trend_6h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below R3 (mean reversion)
            if close[i] < r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above S3 (mean reversion)
            if close[i] > s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot R3/S3 breakout with volume confirmation and weekly trend filter on 6h timeframe.
# Uses weekly high/low/close derived from daily data to calculate pivot levels.
# Weekly trend filter ensures alignment with higher timeframe direction.
# Volume spike (2x 20-period MA) confirms breakout strength.
# Position size 0.25 balances risk and return, targeting 15-30 trades/year to avoid fee drag.
# Designed to work in both bull and bear markets by following weekly trend while using mean-reversion exits.
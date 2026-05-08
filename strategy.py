#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Weekly Pivot + Daily Trend + Volume Spike
# Uses weekly Camarilla pivot levels for bias, daily EMA34 for trend filter, and 6h volume spike (>2x 20-period average) for entry.
# Designed to capture continuation of weekly trend with daily confirmation and volume momentum.
# Target: 12-37 trades/year (50-150 total over 4 years). Works in both bull and bear via weekly pivot bias.

name = "6h_WeeklyPivot_DailyTrend_VolumeSpike"
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
    
    # Get weekly data for Camarilla pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R3, R4, S3, S4)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    pivot = (high_w + low_w + close_w) / 3
    range_w = high_w - low_w
    r3 = pivot + range_w * 1.1
    r4 = pivot + range_w * 1.5
    s3 = pivot - range_w * 1.1
    s4 = pivot - range_w * 1.5
    
    # Get daily data for EMA34 trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate 6h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Align daily EMA34 to 6h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find current daily bar's close
        close_daily_current = np.nan
        if not np.isnan(ema34_daily_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                close_daily_current = df_daily.iloc[idx_daily]['close']
        
        if np.isnan(close_daily_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly bias from pivot levels
        weekly_bullish = close[i] > r3_aligned[i]  # Price above R3 = bullish bias
        weekly_bearish = close[i] < s3_aligned[i]  # Price below S3 = bearish bias
        
        # Check conditions
        price_above_ema = close_daily_current > ema34_daily_aligned[i]
        price_below_ema = close_daily_current < ema34_daily_aligned[i]
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: follow weekly bias with daily trend and volume spike
            if weekly_bullish and price_above_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            elif weekly_bearish and price_below_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly bias turns bearish or daily trend fails or volume drops
            if not weekly_bullish or price_below_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly bias turns bullish or daily trend fails or volume drops
            if not weekly_bearish or price_above_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
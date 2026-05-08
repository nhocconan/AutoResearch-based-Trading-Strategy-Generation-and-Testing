#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Point Reversal with Volume Confirmation
# Uses weekly pivot points (PP, R1, R2, S1, S2) calculated from prior week's OHLC.
# In bullish regime (price above weekly PP), look for long when price rejects S1/S2 with volume.
# In bearish regime (price below weekly PP), look for short when price rejects R1/R2 with volume.
# Volume > 1.3x 20-period average confirms rejection strength.
# Weekly pivot points provide institutional support/resistance levels that work in both bull/bear markets.
# Target: 15-35 trades/year (60-140 over 4 years) to stay within optimal range.

name = "6h_WeeklyPivot_Reversal_Volume"
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
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 52:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    # Calculate weekly pivot points: PP = (H + L + C) / 3
    weekly_pivot = np.full(len(weekly_high), np.nan)
    weekly_r1 = np.full(len(weekly_high), np.nan)
    weekly_s1 = np.full(len(weekly_high), np.nan)
    weekly_r2 = np.full(len(weekly_high), np.nan)
    weekly_s2 = np.full(len(weekly_high), np.nan)
    
    for i in range(len(weekly_high)):
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            weekly_pivot[i] = (weekly_high[i] + weekly_low[i] + weekly_close[i]) / 3.0
            weekly_r1[i] = 2 * weekly_pivot[i] - weekly_low[i]
            weekly_s1[i] = 2 * weekly_pivot[i] - weekly_high[i]
            weekly_r2[i] = weekly_pivot[i] + (weekly_high[i] - weekly_low[i])
            weekly_s2[i] = weekly_pivot[i] - (weekly_high[i] - weekly_low[i])
    
    # Get daily data for volume filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_volume = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(daily_volume), np.nan)
    if len(daily_volume) >= 20:
        for i in range(20, len(daily_volume)):
            vol_avg_20_daily[i] = np.mean(daily_volume[i-20:i])
    
    # Align weekly pivot points to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Align daily volume average to 6h timeframe
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.3x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's volume
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                vol_daily_current = df_daily.iloc[idx_daily]['volume']
                vol_filter = vol_daily_current > 1.3 * vol_avg_20_daily_aligned[i]
        
        # Determine price position relative to weekly pivot levels
        price_above_r2 = close[i] > r2_aligned[i]
        price_between_r1_r2 = r1_aligned[i] < close[i] < r2_aligned[i]
        price_between_pp_r1 = pivot_aligned[i] < close[i] < r1_aligned[i]
        price_between_s1_pp = s1_aligned[i] < close[i] < pivot_aligned[i]
        price_between_s2_s1 = s2_aligned[i] < close[i] < s1_aligned[i]
        price_below_s2 = close[i] < s2_aligned[i]
        
        if position == 0:
            # Look for entry: rejection at pivot levels with volume
            # Long when price rejects S1/S2 (bounces up) in bullish regime (above PP)
            long_condition = (
                (close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1]) or  # bounce from S1
                (close[i] > s2_aligned[i] and close[i-1] <= s2_aligned[i-1])     # bounce from S2
            ) and close[i] > pivot_aligned[i] and vol_filter
            
            # Short when price rejects R1/R2 (bounces down) in bearish regime (below PP)
            short_condition = (
                (close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1]) or  # bounce from R1
                (close[i] < r2_aligned[i] and close[i-1] >= r2_aligned[i-1])     # bounce from R2
            ) and close[i] < pivot_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or reaches R1
            if close[i] < s1_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or reaches S1
            if close[i] > r1_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
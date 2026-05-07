#!/usr/bin/env python3
# 1D_Camarilla_R3S3_WeeklyTrend_VolumeFilter
# Hypothesis: Uses daily Camarilla R3/S3 levels with weekly trend filter (EMA34) and volume spike confirmation.
# Enters long when price breaks above R3 in weekly uptrend with volume confirmation, short when breaks below S3 in weekly downtrend.
# Exits when price returns to the Camarilla mid-point (P) or breaks opposite level.
# Designed for 1d timeframe to target 15-25 trades/year, reducing fee drag while capturing trends in bull/bear markets.

name = "1D_Camarilla_R3S3_WeeklyTrend_VolumeFilter"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema34)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Handle first day where shift creates NaN
    prev_close[0] = close[0]  # Use current close as fallback for first day
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels: R3 and S3
    range_1d = prev_high - prev_low
    r3 = prev_close + range_1d * 1.1 / 4
    s3 = prev_close - range_1d * 1.1 / 4
    pp = (prev_high + prev_low + prev_close) / 3  # Pivot point
    
    # Align to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    pp_aligned = align_htf_to_ltf(prices, df_daily, pp)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for weekly EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(weekly_ema34_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 + weekly uptrend (price > weekly EMA34) + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > weekly_ema34_aligned[i] and   # Weekly uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + weekly downtrend (price < weekly EMA34) + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < weekly_ema34_aligned[i] and   # Weekly downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to pivot point (mean reversion)
            # 2. Opposite Camarilla level break (trend exhaustion)
            at_pivot = abs(close[i] - pp_aligned[i]) < (r3_aligned[i] - pp_aligned[i]) * 0.1  # Within 10% of PP
            opposite_break = (position == 1 and close[i] < s3_aligned[i]) or \
                           (position == -1 and close[i] > r3_aligned[i])
            
            if at_pivot or opposite_break:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals
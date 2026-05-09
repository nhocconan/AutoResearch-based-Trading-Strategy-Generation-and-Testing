#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R3_S3_Breakout_WeekTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    # Use previous week's data to avoid look-ahead
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points (using previous week's data)
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3
    r3_weekly = close_weekly + (high_weekly - low_weekly) * 1.1  # R3 = Close + 1.1*(High-Low)
    s3_weekly = close_weekly - (high_weekly - low_weekly) * 1.1  # S3 = Close - 1.1*(High-Low)
    
    # Align weekly data to daily timeframe (already shifted by one week in get_htf_data)
    pp_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r3_weekly)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s3_weekly)
    
    # Weekly trend filter: EMA(34) on weekly close
    close_weekly_series = pd.Series(close_weekly)
    ema34_weekly = close_weekly_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_weekly_aligned[i]) or np.isnan(r3_weekly_aligned[i]) or 
            np.isnan(s3_weekly_aligned[i]) or np.isnan(ema34_weekly_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Close breaks above R3 with volume spike and above weekly EMA trend
            if close[i] > r3_weekly_aligned[i] and vol_ok and close[i] > ema34_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 with volume spike and below weekly EMA trend
            elif close[i] < s3_weekly_aligned[i] and vol_ok and close[i] < ema34_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S3 (mean reversion)
            if close[i] < s3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back above R3
            if close[i] > r3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
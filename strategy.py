#!/usr/bin/env python3
"""
Hypothesis: Weekly pivot point S1/S3 breakout on daily chart with weekly EMA34 trend filter and volume confirmation.
Long when daily close breaks above weekly S3 with bullish weekly trend and volume spike.
Short when daily close breaks below weekly S1 with bearish weekly trend and volume spike.
Exit when price returns to weekly pivot or trend reverses.
Designed for low trade frequency (10-25/year) to minimize fee drift on 1d timeframe.
Works in bull markets via breakout momentum and bear markets via mean reversion at extremes.
"""
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
    
    # Load weekly data for pivot and trend - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_w = pd.Series(df_weekly['close'].values)
    ema34_w = close_w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to daily timeframe
    ema34_w_aligned = align_htf_to_ltf(prices, df_weekly, ema34_w)
    
    # Calculate weekly pivot levels from previous week's OHLC
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate pivot and levels for each week
    pivot_w = (high_w + low_w + close_w) / 3.0
    range_w = high_w - low_w
    
    # Weekly Camarilla levels (using weekly range)
    s1_w = close_w - (range_w * 1.1 / 12)
    s3_w = close_w - (range_w * 1.1 / 4)
    r1_w = close_w + (range_w * 1.1 / 12)
    r3_w = close_w + (range_w * 1.1 / 4)
    pivot_w = (high_w + low_w + close_w) / 3.0
    
    # Align all levels to daily timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_weekly, pivot_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_weekly, s3_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_weekly, r3_w)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC) - though less critical on 1d
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after weekly lookback
        # Skip if data not ready
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(s3_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(r3_w_aligned[i]) or np.isnan(ema34_w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC (optional on 1d, but keeps consistency)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Daily close breaks above weekly S3 with bullish weekly trend and volume spike
            if (close[i] > s3_w_aligned[i] and 
                close[i] > ema34_w_aligned[i] and  # Bullish trend: price above weekly EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Daily close breaks below weekly S1 with bearish weekly trend and volume spike
            elif (close[i] < s1_w_aligned[i] and 
                  close[i] < ema34_w_aligned[i] and  # Bearish trend: price below weekly EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to weekly pivot OR trend turns bearish
                if close[i] <= pivot_w_aligned[i] or close[i] < ema34_w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to weekly pivot OR trend turns bullish
                if close[i] >= pivot_w_aligned[i] or close[i] > ema34_w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyPivot_S1S3_WeeklyEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%
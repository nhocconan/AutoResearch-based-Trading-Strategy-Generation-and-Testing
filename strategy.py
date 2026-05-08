#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume confirmation and 12h EMA trend filter
# Uses daily Camarilla levels (R3/S3) for entries when price breaks these levels,
# confirmed by daily volume > 1.2x 20-day EMA and 12h EMA50 trend direction.
# Exits when price returns to the 12h EMA50 or when volume drops below average.
# Designed to capture strong trending moves with minimal trades.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dVolume_12hTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for Camarilla and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's range)
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    prev_close = df_daily['close'].values[:-1]
    prev_high = df_daily['high'].values[:-1]
    prev_low = df_daily['low'].values[:-1]
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Calculate daily volume EMA (20-period)
    vol_ema_20 = pd.Series(df_daily['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ema_20)
    
    # Align 12h EMA to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for NaN values
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(vol_ema_20_aligned[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.2x 20-day EMA
        # Find the most recent completed daily bar
        idx_daily = 0
        while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
            idx_daily += 1
        idx_daily -= 1  # last completed daily bar
        
        if idx_daily < 0:
            vol_filter = False
        else:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.2 * vol_ema_20_aligned[i]
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Look for breakout entry with volume and trend confirmation
            if close[i] > camarilla_r3_aligned[i] and vol_filter and trend_up:
                signals[i] = 0.25
                position = 1
            elif close[i] < camarilla_s3_aligned[i] and vol_filter and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 12h EMA50 or volume filter fails
            if close[i] <= ema_50_12h_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 12h EMA50 or volume filter fails
            if close[i] >= ema_50_12h_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
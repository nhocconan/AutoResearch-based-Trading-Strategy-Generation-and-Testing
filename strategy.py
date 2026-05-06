#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-weekly pivot points with volume expansion and EMA trend filter
# Long when price breaks above weekly R3 with volume > 2x 20-period average and EMA50 > EMA200 (bullish trend)
# Short when price breaks below weekly S3 with volume > 2x 20-period average and EMA50 < EMA200 (bearish trend)
# Uses weekly pivot points for key support/resistance, volume for confirmation, and EMA for trend filter
# Designed to capture strong breakouts in trending markets while avoiding false signals in ranging markets
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing

name = "6h_WeeklyPivot_R3S3_Breakout_VolumeTrend"
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
    
    # Calculate weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly high, low, close for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # EMA trend filter (50 and 200)
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).values
    
    # Volume confirmation: >2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R3 with volume confirmation and bullish trend
            if close[i] > r3_aligned[i] and volume_filter[i] and ema_50[i] > ema_200[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly S3 with volume confirmation and bearish trend
            elif close[i] < s3_aligned[i] and volume_filter[i] and ema_50[i] < ema_200[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S3 (support break) or trend turns bearish
            if close[i] < s3_aligned[i] or ema_50[i] < ema_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R3 (resistance break) or trend turns bullish
            if close[i] > r3_aligned[i] or ema_50[i] > ema_200[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
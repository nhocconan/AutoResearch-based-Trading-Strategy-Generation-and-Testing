#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 12h EMA50 trend filter and volume confirmation.
# Long when price touches S1 level in uptrend (price > EMA50) with volume spike.
# Short when price touches R1 level in downtrend (price < EMA50) with volume spike.
# Exit when price moves to opposite pivot level (S3 for long, R3 for short) or returns to pivot point.
# Uses Camarilla levels from daily timeframe, EMA50 from 12h for trend, volume > 2x 20-period average.
# Designed for mean reversion in ranging markets with trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year to minimize fee drag while maintaining edge.

name = "4h_Camarilla_Pivot_Reversal_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # S1 = C - (Range * 1.1/12), S2 = C - (Range * 1.1/6), S3 = C - (Range * 1.1/4)
    # R1 = C + (Range * 1.1/12), R2 = C + (Range * 1.1/6), R3 = C + (Range * 1.1/4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    pp = np.full(len(df_1d), np.nan)
    s1 = np.full(len(df_1d), np.nan)
    s2 = np.full(len(df_1d), np.nan)
    s3 = np.full(len(df_1d), np.nan)
    r1 = np.full(len(df_1d), np.nan)
    r2 = np.full(len(df_1d), np.nan)
    r3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        pp[i] = (h + l + c) / 3.0
        rng = h - l
        s1[i] = c - (rng * 1.1 / 12)
        s2[i] = c - (rng * 1.1 / 6)
        s3[i] = c - (rng * 1.1 / 4)
        r1[i] = c + (rng * 1.1 / 12)
        r2[i] = c + (rng * 1.1 / 6)
        r3[i] = c + (rng * 1.1 / 4)
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period average volume for volume filter (using 12h data)
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(ema_50_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 2x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla touch + trend + volume
            # Long when price touches S1 in uptrend (price > EMA50) with volume spike
            long_condition = (abs(close[i] - s1_aligned[i]) < 0.001 * close[i]) and \
                           (close[i] > ema_50_aligned[i]) and vol_filter
            # Short when price touches R1 in downtrend (price < EMA50) with volume spike
            short_condition = (abs(close[i] - r1_aligned[i]) < 0.001 * close[i]) and \
                            (close[i] < ema_50_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches S3 (strong support) or returns to pivot point
            if close[i] <= s3_aligned[i] or close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches R3 (strong resistance) or returns to pivot point
            if close[i] >= r3_aligned[i] or close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 6h_Pivot_Reversal_12hTrend_VolumeFilter
# Hypothesis: Fade at daily Camarilla R3/S3 levels when 12h trend is strong, with volume confirmation.
# Works in bull/bear by fading extremes only when aligned with higher timeframe trend.
# Targets 20-40 trades/year to minimize fee drag.

name = "6h_Pivot_Reversal_12hTrend_VolumeFilter"
timeframe = "6h"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+C)/3 (typical price)
    # We'll use previous day's data to avoid look-ahead
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate typical price for previous day
    prev_daily_typical = (daily_high[:-1] + daily_low[:-1] + daily_close[:-1]) / 3
    prev_daily_range = daily_high[:-1] - daily_low[:-1]
    
    # Calculate Camarilla levels for previous day
    r3 = prev_daily_typical + prev_daily_range * 1.1 / 4
    s3 = prev_daily_typical - prev_daily_range * 1.1 / 4
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 6s timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20) + 1  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend filter
        uptrend = close[i] > ema_20_12h_aligned[i]
        downtrend = close[i] < ema_20_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price at S3 support + uptrend + volume spike
            if close[i] <= s3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price at R3 resistance + downtrend + volume spike
            elif close[i] >= r3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price moves back above S3 or trend changes
            if close[i] > s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price moves back below R3 or trend changes
            if close[i] < r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
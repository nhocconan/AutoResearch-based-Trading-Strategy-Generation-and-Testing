#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1w data once for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly Camarilla calculation
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    
    # Weekly Camarilla pivot levels (R3/S3 are stronger levels)
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    range_val = prev_week_high - prev_week_low
    r3 = pivot + (range_val * 1.1 / 2)  # R3 level
    s3 = pivot - (range_val * 1.1 / 2)  # S3 level
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # 1d trend filter: EMA34 > EMA89 for uptrend, < for downtrend
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89 = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    daily_trend = ema_34 > ema_89  # True for uptrend, False for downtrend
    
    # Align daily trend to 6h timeframe
    daily_trend_6h = align_htf_to_ltf(prices, df_1d, daily_trend.astype(float))
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 89  # warmup for EMA89
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(daily_trend_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R3 with volume spike and daily uptrend
            long_cond = (close[i] > r3_6h[i] and vol_spike[i] and daily_trend_6h[i] > 0.5)
            
            # Short entry: price breaks below weekly S3 with volume spike and daily downtrend
            short_cond = (close[i] < s3_6h[i] and vol_spike[i] and daily_trend_6h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S3 (strong reversal)
            if close[i] < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above weekly R3 (strong reversal)
            if close[i] > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla R3/S3 breakout strategy with volume spike confirmation and daily trend filter on 6h timeframe.
# Uses WEEKLY pivot levels (not daily) for stronger support/resistance. 
# Enters long when price breaks above weekly R3 with volume spike and daily uptrend (EMA34 > EMA89).
# Enters short when price breaks below weekly S3 with volume spike and daily downtrend (EMA34 < EMA89).
# Exits when price breaks through the opposing S3/R3 level (strong reversal signal).
# Weekly timeframe provides structural levels that are more significant than daily, reducing false breakouts.
# Volume spike confirms institutional participation. Daily trend filter ensures alignment with intermediate trend.
# Targets 20-40 trades/year on 6h timeframe (80-160 total over 4 years).
# Works in bull markets (trend-following breakouts from weekly resistance) and bear markets 
# (breakdowns from weekly support with trend alignment).
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    r1 = pivot + (range_val * 1.1 / 6)
    s1 = pivot - (range_val * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d trend: close > open (bullish day)
    daily_bullish = df_1d['close'] > df_1d['open']
    daily_trend_4h = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    
    # 1w trend: weekly close > weekly open (bullish week)
    prev_week_open = np.roll(df_1w['open'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_open[0] = df_1w['open'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    weekly_bullish = prev_week_close > prev_week_open
    weekly_trend_4h = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(daily_trend_4h[i]) or np.isnan(weekly_trend_4h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and daily/weekly uptrend
            long_cond = (close[i] > r3_4h[i] and vol_spike[i] and daily_trend_4h[i] > 0.5 and weekly_trend_4h[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and daily/weekly downtrend
            short_cond = (close[i] < s3_4h[i] and vol_spike[i] and daily_trend_4h[i] < 0.5 and weekly_trend_4h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below R1 (reversal signal)
            if close[i] < r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above S1 (reversal signal)
            if close[i] > s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout strategy with volume spike confirmation and daily/weekly trend filters on 4h timeframe.
# Enters long when price breaks above R3 with volume spike and both daily and weekly bullish trends.
# Enters short when price breaks below S3 with volume spike and both daily and weekly bearish trends.
# Exits when price reverses back through R1/S1 respectively.
# Uses multiple timeframe confirmation (1d/1w) to filter trades and reduce whipsaw.
# Uses discrete sizing (0.25) to minimize churn. Targets 20-40 trades/year on 4h timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).
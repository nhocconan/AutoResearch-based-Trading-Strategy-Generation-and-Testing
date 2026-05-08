#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data once for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily trend: close > EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_trend = close_1d > ema_34_1d  # True for uptrend, False for downtrend
    daily_trend_4h = align_htf_to_ltf(prices, df_1d, daily_trend.astype(float))
    
    # Weekly trend: close > EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_trend = close_1w > ema_34_1w  # True for uptrend, False for downtrend
    weekly_trend_4h = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # Previous day's OHLC for 4h Camarilla calculation
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 6)
    s1 = pivot - (range_val * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(daily_trend_4h[i]) or 
            np.isnan(weekly_trend_4h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and both daily and weekly uptrend
            long_cond = (close[i] > r1_4h[i] and vol_spike[i] and 
                        daily_trend_4h[i] > 0.5 and weekly_trend_4h[i] > 0.5)
            
            # Short entry: price breaks below S1 with volume spike and both daily and weekly downtrend
            short_cond = (close[i] < s1_4h[i] and vol_spike[i] and 
                         daily_trend_4h[i] < 0.5 and weekly_trend_4h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 (reversal signal)
            if close[i] < s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above R1 (reversal signal)
            if close[i] > r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout strategy with volume spike confirmation and dual timeframe trend filter (daily + weekly) on 4h timeframe.
# Enters long when price breaks above R1 with volume spike and both daily and weekly uptrend (close > EMA34).
# Enters short when price breaks below S1 with volume spike and both daily and weekly downtrend (close < EMA34).
# Exits when price reverses back through S1/R1 respectively.
# Uses dual timeframe trend filter to ensure alignment with higher timeframe trends, reducing whipsaw in sideways markets.
# Uses discrete sizing (0.25) to minimize churn. Targets 20-40 trades/year on 4h timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).
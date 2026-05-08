#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1d data once for trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for daily trend (yesterday's close > yesterday's open)
    prev_day_open = np.roll(df_1d['open'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    prev_day_open[0] = df_1d['open'].values[0]
    prev_day_close[0] = df_1d['close'].values[0]
    daily_trend = prev_day_close > prev_day_open  # True for uptrend, False for downtrend
    
    # Align daily trend to 4h timeframe
    daily_trend_4h = align_htf_to_ltf(prices, df_1d, daily_trend.astype(float))
    
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
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(daily_trend_4h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike and daily uptrend
            long_cond = (close[i] > r1_4h[i] and vol_spike[i] and daily_trend_4h[i] > 0.5)
            
            # Short entry: price breaks below S1 with volume spike and daily downtrend
            short_cond = (close[i] < s1_4h[i] and vol_spike[i] and daily_trend_4h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses back below S1 (reversal signal)
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

# Hypothesis: Camarilla R1/S1 breakout strategy with volume spike confirmation and daily trend filter on 4h timeframe.
# Enters long when price breaks above R1 with volume spike and daily uptrend (yesterday's close > yesterday's open).
# Enters short when price breaks below S1 with volume spike and daily downtrend (yesterday's close < yesterday's open).
# Exits when price reverses back through S1/R1 respectively.
# Uses daily trend filter to align with higher timeframe momentum, reducing whipsaw in sideways markets.
# Volume spike ensures participation during active market conditions.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).
# Discrete sizing (0.25) minimizes churn. Targets ~20-40 trades/year on 4h timeframe.
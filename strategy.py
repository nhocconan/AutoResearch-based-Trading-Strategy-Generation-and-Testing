#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume confirmation.
# Long when price closes above R1 with 4h EMA50 uptrend (close > EMA50) and 1d volume > 1.5x 20-bar average.
# Short when price closes below S1 with 4h EMA50 downtrend (close < EMA50) and 1d volume > 1.5x average.
# Exit when price reverses and closes below/above the opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses discrete position sizing 0.20. Target: 60-150 total trades over 4 years on 1h timeframe.
# 1d volume filter ensures we only trade on institutional participation, reducing false breakouts.
# 4h EMA50 ensures we only trade in the direction of the intermediate trend.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_1dVolume_Trend_v1"
timeframe = "1h"
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
    
    # Calculate Camarilla levels from previous day (approx using 24x 1h bars)
    lookback = 24  # 24 * 1h = 24h approx
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous "day"
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla R1 and S1 levels
    camarilla_range = high_prev - low_prev
    r1 = close_prev + 1.1 * camarilla_range / 4
    s1 = close_prev - 1.1 * camarilla_range / 4
    
    # Calculate average volume for confirmation (20-period) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 4h EMA50 to 1h timeframe (wait for 4h bar to close)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above R1 with 4h EMA50 uptrend and 1d volume > 1.5x average
            if (close[i] > r1[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                vol_1d[i] > 1.5 * avg_volume_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price closes below S1 with 4h EMA50 downtrend and 1d volume > 1.5x average
            elif (close[i] < s1[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  vol_1d[i] > 1.5 * avg_volume_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversal signal)
            if close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversal signal)
            if close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
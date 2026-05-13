#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R4/S4 breakout with weekly trend filter and volume spike confirmation.
# Long when price breaks above R4 with weekly close above weekly open (bullish weekly candle) and volume > 2.0x average.
# Short when price breaks below S4 with weekly close below weekly open (bearish weekly candle) and volume > 2.0x average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Weekly trend filter ensures we trade with the dominant higher timeframe momentum, reducing whipsaw.
# Volume spike confirms institutional participation. Works in bull markets via upward breaks and in bear markets via downward breaks.

name = "6h_Camarilla_R4_S4_Breakout_1wTrend_VolumeSpike_v1"
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
    
    # Calculate Camarilla levels from previous day (approx using 4x 6h bars)
    lookback = 4  # 4 * 6h = 24h approx
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for previous "day"
    high_prev = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    low_prev = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    close_prev = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    # Camarilla R4 and S4 levels (breakout levels)
    camarilla_range = high_prev - low_prev
    r4 = close_prev + 1.1 * camarilla_range
    s4 = close_prev - 1.1 * camarilla_range
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    
    # Weekly bullish/bearish candle: 1 if bullish (close > open), -1 if bearish (close < open)
    weekly_trend_raw = np.where(close_1w > open_1w, 1, -1)
    
    # Align weekly trend to 6h timeframe (wait for weekly bar to close)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R4 with bullish weekly candle and volume spike
            if (close[i] > r4[i] and 
                weekly_trend_aligned[i] == 1 and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S4 with bearish weekly candle and volume spike
            elif (close[i] < s4[i] and 
                  weekly_trend_aligned[i] == -1 and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S4 (reversal signal)
            if close[i] < s4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R4 (reversal signal)
            if close[i] > r4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
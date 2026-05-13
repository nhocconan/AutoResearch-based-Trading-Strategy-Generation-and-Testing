#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1w EMA200 trend filter and volume spike > 2.0x average.
# Long when price closes above R3 with 1w EMA200 uptrend (close > EMA200) and volume > 2.0x 20-bar average volume.
# Short when price closes below S3 with 1w EMA200 downtrend (close < EMA200) and volume > 2.0x average.
# Exit when price reverses and closes below/above the opposite Camarilla level (S3 for longs, R3 for shorts).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# Higher volume threshold (2.0x vs 1.8x) reduces overtrading and fee drag while maintaining edge in strong moves.
# 1w EMA200 ensures we only trade in the direction of the long-term trend, avoiding counter-trend false breakouts.
# Weekly timeframe provides robust trend filter that works in both bull and bear markets.

name = "6h_Camarilla_R3_S3_Breakout_1wEMA200_Trend_VolumeSpike_v1"
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
    
    # Camarilla R3 and S3 levels
    camarilla_range = high_prev - low_prev
    r3 = close_prev + 1.1 * camarilla_range / 2
    s3 = close_prev - 1.1 * camarilla_range / 2
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA200 on 1w data
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1w EMA200 to 6h timeframe (wait for 1w bar to close)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback + 20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price closes above R3 with 1w EMA200 uptrend and volume spike > 2.0x
            if (close[i] > r3[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S3 with 1w EMA200 downtrend and volume spike > 2.0x
            elif (close[i] < s3[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 (reversal signal)
            if close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 (reversal signal)
            if close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R3/S3 breakout with 1w EMA20 trend filter and volume confirmation
# Camarilla pivot levels act as intraday support/resistance; breakouts above R3 or below S3
# indicate strong momentum. Combined with 1w EMA20 trend filter (bullish if price > EMA20,
# bearish if price < EMA20) and volume confirmation (1.5x 20-period EMA), this strategy
# captures explosive moves aligned with the weekly trend. Designed for 12h timeframe to
# target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying R3 breakouts in uptrends and in bear markets by selling
# S3 breakdowns in downtrends, avoiding false breakouts in ranging conditions.

name = "12h_Camarilla_R3S3_Breakout_1wEMA20_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    
    # Align 1d Camarilla levels to 12h timeframe (using previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 1.5x 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 + volume confirmation + price above 1w EMA20 (uptrend)
            if (close[i] > r3_aligned[i] and volume_confirmed and 
                close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume confirmation + price below 1w EMA20 (downtrend)
            elif (close[i] < s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below R3 (failed breakout) OR price below 1w EMA20 (trend change)
            if close[i] < r3_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above S3 (failed breakdown) OR price above 1w EMA20 (trend change)
            if close[i] > s3_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
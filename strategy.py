#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Camarilla pivot levels provide intraday support/resistance. Breakouts above R3 or below S3
# with volume confirmation and aligned with 4h trend (EMA50) capture momentum moves.
# Designed for 15-37 trades/year on 1h to minimize fee drag. Works in bull markets via
# long breakouts above R3 with bullish 4h trend, and in bear markets via short breakouts
# below S3 with bearish 4h trend.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_prev = np.roll(df_1d['high'].values, 1)
    low_prev = np.roll(df_1d['low'].values, 1)
    close_prev = np.roll(df_1d['close'].values, 1)
    # Set first day's previous values to NaN (no prior day)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels: R3, S3
    r3 = close_prev + (range_prev * 1.1 / 4)
    s3 = close_prev - (range_prev * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND above 4h EMA50 AND volume spike AND in session
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i] and 
                in_session[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 AND below 4h EMA50 AND volume spike AND in session
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i] and 
                  in_session[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below R3 OR below 4h EMA50
            if close[i] < r3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above S3 OR above 4h EMA50
            if close[i] > s3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
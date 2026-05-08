#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume spike
# Uses Alligator jaws (13-period SMMA) and teeth (8-period SMMA) for trend.
# Entry when lips (5-period SMMA) crosses jaws with trend alignment and volume.
# Designed for 15-25 trades/year on 12h timeframe.
name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    sma = np.mean(data[:period])
    result[period-1] = sma
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Williams Alligator components on 12h data
    # Jaws (13-period SMMA of median price)
    median_price = (high + low) / 2
    jaws = smma(median_price, 13)
    # Teeth (8-period SMMA of median price)
    teeth = smma(median_price, 8)
    # Lips (5-period SMMA of median price)
    lips = smma(median_price, 5)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(lips[i]) or np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: lips crosses above jaws with trend alignment and volume spike
            if (lips[i] > jaws[i] and lips[i-1] <= jaws[i-1] and
                close[i] > ema34_1d_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short entry: lips crosses below jaws with trend alignment and volume spike
            elif (lips[i] < jaws[i] and lips[i-1] >= jaws[i-1] and
                  close[i] < ema34_1d_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: lips crosses below jaws (trend change)
            if lips[i] < jaws[i] and lips[i-1] >= jaws[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: lips crosses above jaws (trend change)
            if lips[i] > jaws[i] and lips[i-1] <= jaws[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
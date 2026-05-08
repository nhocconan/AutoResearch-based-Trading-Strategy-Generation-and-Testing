#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume spike.
# Williams %R > -20 indicates overbought, < -80 indicates oversold.
# In 1d uptrend: look for pullbacks to oversold (-80) for long entries.
# In 1d downtrend: look for bounces from overbought (-20) for short entries.
# Volume spike (2.0x 20-period EMA) confirms momentum at turning points.
# Target: 60-100 total trades over 4 years (15-25/year) to minimize fee drag.

name = "6h_WilliamsR_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams %R (14-period) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1d EMA(20) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up = ema_20_1d[1:] > ema_20_1d[:-1]  # Rising EMA = uptrend
    trend_up = np.concatenate([[False], trend_up])  # Align with 1d index
    
    # Volume confirmation: 20-period volume spike (2.0x EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    # Align 1d indicators to 6h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, lookback)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: 1d uptrend + Williams %R oversold + volume spike
            if (trend_up_aligned[i] > 0.5 and      # 1d uptrend
                williams_r[i] <= -80 and           # Oversold
                vol_confirm[i]):                   # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short entry: 1d downtrend + Williams %R overbought + volume spike
            elif (trend_up_aligned[i] <= 0.5 and   # 1d downtrend
                  williams_r[i] >= -20 and         # Overbought
                  vol_confirm[i]):                 # Volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: reverse signal or overbought condition
            if (trend_up_aligned[i] <= 0.5 or      # 1d trend turned down
                williams_r[i] >= -20):             # Overbought - take profit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: reverse signal or oversold condition
            if (trend_up_aligned[i] > 0.5 or       # 1d trend turned up
                williams_r[i] <= -80):             # Oversold - take profit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
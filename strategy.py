#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: 34 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    daily_uptrend = close > ema_34_1d_aligned
    
    # Volume spike (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma20
    
    # Camarilla levels from previous day
    # Calculate using previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(close, 1)  # approximation for 12h timeframe
    
    # First bar needs special handling
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = close[0]
    
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(daily_uptrend[i]) or np.isnan(volume_spike[i]) or np.isnan(R3[i]) or np.isnan(S3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and daily uptrend
            if close[i] > R3[i] and volume_spike[i] and daily_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and daily downtrend
            elif close[i] < S3[i] and volume_spike[i] and not daily_uptrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below R3 or daily trend changes
            if close[i] < R3[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above S3 or daily trend changes
            if close[i] > S3[i] or daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
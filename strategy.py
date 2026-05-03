#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation
# Using 1d EMA200 as trend filter to avoid whipsaws in ranging markets, combined with
# Camarilla breakouts on 1h for precise entry timing. Volume spike confirms institutional
# participation. Designed for 15-30 trades/year on 1h timeframe to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (failed breaks reverse).

name = "1h_Camarilla_R3S3_1dEMA200_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Camarilla levels: based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1
    r3 = close_1d + camarilla_range / 4
    s3 = close_1d - camarilla_range / 4
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 1h data for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after sufficient warmup for EMA200
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks R3 or S3 with volume spike
        breakout_long = close[i] > r3_aligned[i] and volume_spike[i]
        breakout_short = close[i] < s3_aligned[i] and volume_spike[i]
        
        if position == 0:
            # Long: break above R3 in 1d uptrend with volume spike
            if breakout_long and ema_200_1d_aligned[i] < close[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below S3 in 1d downtrend with volume spike
            elif breakout_short and ema_200_1d_aligned[i] > close[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below R3 or loses 1d uptrend
            if close[i] < r3_aligned[i] or ema_200_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above S3 or loses 1d downtrend
            if close[i] > s3_aligned[i] or ema_200_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
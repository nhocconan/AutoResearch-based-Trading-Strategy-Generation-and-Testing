#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot with volume spike and 1d trend filter on 1h timeframe.
# Uses 4h Camarilla levels (R3/S3) for breakout signals, 4h volume spike for confirmation,
# and 1d EMA(34) for trend direction. Long when price breaks above R3 with volume spike and 1d uptrend.
# Short when price breaks below S3 with volume spike and 1d downtrend.
# Operates only during 08-20 UTC to avoid low-volume sessions.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_Camarilla_R3S3_Volume_1dTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # R3 = High + 2*(High - Low), S3 = Low - 2*(High - Low)
    # Using previous 4h bar's range
    range_4h = high_4h - low_4h
    r3_4h = high_4h + 2 * range_4h
    s3_4h = low_4h - 2 * range_4h
    
    # Align Camarilla levels to 1h
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # 4h volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.5 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike.astype(float))
    
    # 1d EMA(34) for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and 1d uptrend
            if (close[i] > r3_4h_aligned[i] and 
                volume_spike_aligned[i] > 0.5 and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume spike and 1d downtrend
            elif (close[i] < s3_4h_aligned[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA or reverse signal
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above 1d EMA or reverse signal
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
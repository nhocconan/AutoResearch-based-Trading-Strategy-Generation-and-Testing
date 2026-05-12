#!/usr/bin/env python3
name = "1h_R3S3_Breakout_4hTrend_1dVolatility"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # precomputed DatetimeIndex.hour
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 1d data once for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Daily ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d_vals[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_vals[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily pivot for Camarilla levels (previous day)
    p = (high_1d + low_1d + close_1d_vals) / 3
    r3 = close_1d_vals + (high_1d - low_1d) * 1.1 / 4
    s3 = close_1d_vals - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 1h (wait for daily close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if position == 0:
            # Long: price breaks above R3 + above 4h EMA20 + volatility filter + session
            if (in_session and 
                close[i] > r3_aligned[i] and 
                close[i] > ema_20_4h_aligned[i] and 
                atr_14_1d_aligned[i] > 0):  # volatility present
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 + below 4h EMA20 + volatility filter + session
            elif (in_session and 
                  close[i] < s3_aligned[i] and 
                  close[i] < ema_20_4h_aligned[i] and 
                  atr_14_1d_aligned[i] > 0):  # volatility present
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S3
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R3
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_1dVolume"
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
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h Camarilla R3/S3 levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    range_4h = high_4h - low_4h
    r3_4h = close_4h + 1.1666 * range_4h * 1.1 / 2
    s3_4h = close_4h - 1.1666 * range_4h * 1.1 / 2
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + 4h uptrend + high 1d volume
            if close[i] > r3_4h_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > vol_avg_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 + 4h downtrend + high 1d volume
            elif close[i] < s3_4h_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > vol_avg_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price crosses back through the opposite S3/R3 level
            if position == 1:
                if close[i] < s3_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > r3_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals
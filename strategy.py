#!/usr/bin/env python3
name = "1h_MultiTF_Trend_With_Volume"
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
    
    # Pre-compute hourly time for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    # 1d volume average
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1h volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)  # UTC 8-20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: price vs 4h EMA50
        trend_up = close[i] > ema_4h_aligned[i]
        
        # Volume condition: 1h volume > 1.5x 20-ma AND 1d volume > 1.2x its average
        vol_cond = (volume[i] > 1.5 * vol_ma20[i]) and \
                   (volume_1d[i//24] > 1.2 * vol_avg_1d_aligned[i] if i//24 < len(volume_1d) else False)
        
        if position == 0:
            # Long: uptrend + volume confirmation
            if trend_up and vol_cond:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + volume confirmation
            elif not trend_up and vol_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend reversal or volume drops
            if not trend_up or not vol_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend reversal or volume drops
            if trend_up or not vol_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
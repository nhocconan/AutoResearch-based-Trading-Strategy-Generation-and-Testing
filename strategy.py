#!/usr/bin/env python3
"""
1h Volume Spike + 4h EMA Trend + Session Filter
Hypothesis: On 1h, volume spikes (>2x 20-period average) combined with 4h EMA50 trend alignment
and UTC 08-20 session filter capture institutional participation while avoiding low-liquidity periods.
The 4h EMA provides the directional bias, reducing whipsaws. Volume spike confirms momentum.
Works in bull/bear by following 4h trend. Target: 60-150 total trades over 4 years.
"""

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
    
    # Pre-compute session hours (08-20 UTC) - prices.index is DatetimeIndex
    session_hours = prices.index.hour
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    # 4h EMA50 for trend filter - loaded ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50 + VolMA20
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_50_level = ema_50_4h_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Exit conditions: opposite volume spike or trend change
        if position != 0:
            if position == 1 and (curr_close < ema_50_level or 
                                  (volume_spike and curr_close < close[i-1])):
                signals[i] = 0.0
                position = 0
                continue
            elif position == -1 and (curr_close > ema_50_level or 
                                     (volume_spike and curr_close > close[i-1])):
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Volume spike + trend alignment
        if position == 0:
            long_condition = volume_spike and (curr_close > ema_50_level)
            short_condition = volume_spike and (curr_close < ema_50_level)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
    
    return signals

name = "1h_VolumeSpike_4hEMA50_Trend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0
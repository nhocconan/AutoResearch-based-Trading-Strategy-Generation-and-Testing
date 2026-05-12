#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
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
    
    # 4h trend: EMA(34) on 4h close
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 1h session filter: 08-20 UTC
    hours = prices.index.hour
    
    # 1d volume spike: current volume > 2.0x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    # 1d OHLC for Camarilla R1 and S1 (previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    p = (high_1d + low_1d + close_1d_vals) / 3
    r1 = close_1d_vals + (high_1d - low_1d) * 1.1 / 2
    s1 = close_1d_vals - (high_1d - low_1d) * 1.1 / 2
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above R1 + 4h trend up + 1d volume spike + session
            if (close[i] > r1_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and 
                vol_spike[i] and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h trend down + 1d volume spike + session
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and 
                  vol_spike[i] and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S1
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
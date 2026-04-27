#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Breakout at Camarilla R1/S1 levels with 4h trend filter (EMA50) and 1d volume spike.
Long: price breaks above R1 + 4h uptrend + 1d volume > 1.5x 20-day avg
Short: price breaks below S1 + 4h downtrend + 1d volume > 1.5x 20-day avg
Exit: price returns to Camarilla Pivot point or trend fails.
Uses 1h for entry timing, 4h for trend direction, 1d for volume filter.
Target: 20-40 trades/year per symbol.
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
    
    # Calculate Camarilla levels from previous day
    # Need daily high, low, close - get from 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    P = (prev_high + prev_low + prev_close) / 3  # Pivot point
    
    # Align to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d volume spike filter: current volume > 1.5x 20-day average
    vol_20avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = df_1d['volume'].values > (vol_20avg * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need Camarilla levels, 4h EMA, 1d volume spike
    start_idx = max(20, 1)  # need 20-day vol avg and previous day
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(P_aligned[i]) or np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        p = P_aligned[i]
        ema4h = ema_4h_aligned[i]
        vol_spike_ok = vol_spike_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume spike
            if price > r1 and price > ema4h and vol_spike_ok:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume spike
            elif price < s1 and price < ema4h and vol_spike_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or trend fails
            if price < p or price < ema4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or trend fails
            if price > p or price > ema4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0
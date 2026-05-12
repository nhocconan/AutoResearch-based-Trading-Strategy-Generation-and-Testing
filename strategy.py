#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Camarilla pivot levels: calculate once per 4h bar
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    camarilla_S1 = close_4h - 1.0833 * (high_4h - low_4h) / 12
    camarilla_R1 = close_4h + 1.0833 * (high_4h - low_4h) / 12
    
    # Align Camarilla levels to 1h timeframe with proper delay
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R1)
    
    # 4h EMA50 trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h volume filter: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above R1 + above 4h EMA50 + volume expansion
            if (close[i] > camarilla_R1_aligned[i] and 
                close[i-1] <= camarilla_R1_aligned[i-1] and
                close[i] > ema50_4h_aligned[i] and
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price crosses below S1 + below 4h EMA50 + volume expansion
            elif (close[i] < camarilla_S1_aligned[i] and 
                  close[i-1] >= camarilla_S1_aligned[i-1] and
                  close[i] < ema50_4h_aligned[i] and
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 or below 4h EMA50
            if (close[i] < camarilla_S1_aligned[i] and close[i-1] >= camarilla_S1_aligned[i-1]) or \
               (close[i] < ema50_4h_aligned[i] and close[i-1] >= ema50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above R1 or above 4h EMA50
            if (close[i] > camarilla_R1_aligned[i] and close[i-1] <= camarilla_R1_aligned[i-1]) or \
               (close[i] > ema50_4h_aligned[i] and close[i-1] <= ema50_4h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
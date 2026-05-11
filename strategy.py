#!/usr/bin/env python3
name = "1h_Supertrend_4hTrend_VolumeFilter"
timeframe = "1h"
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
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close)
    atr[atr_period:] = pd.Series(tr[atr_period:]).ewm(alpha=1/atr_period, adjust=False).values
    
    # Calculate Supertrend
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    sma_4h = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    trend_4h = close_4h > sma_4h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(supertrend[i]) or np.isnan(trend_4h_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend + 4h uptrend + volume confirmation
            if direction[i] == 1 and trend_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: Supertrend downtrend + 4h downtrend + volume confirmation
            elif direction[i] == -1 and not trend_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Supertrend downtrend
            if direction[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Supertrend uptrend
            if direction[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
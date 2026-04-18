#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume spike filter and 1h EMA trend filter.
# Camarilla levels provide high-probability reversal/breakout zones based on prior day's range.
# Volume spike confirms institutional participation in the breakout.
# 1h EMA filter ensures trades align with higher timeframe momentum to avoid counter-trend entries.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above H3/L3 with trend) and bear markets (breakouts below L3/H3 with trend).
name = "4h_Camarilla_VolumeSpike_1hEMA_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's range
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels: H4, H3, L3, L4
    range_prev = high_prev - low_prev
    H3 = close_prev + range_prev * 1.1 / 6
    L3 = close_prev - range_prev * 1.1 / 6
    
    # Get 1h data for EMA trend filter
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_1h = pd.Series(close_1h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_1h)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3[i-1]) if i-1 >= 0 else True or np.isnan(L3[i-1]) if i-1 >= 0 else True or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_1h_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume above 1.5x average
        vol_spike = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price breaks above H3 AND volume spike AND price above 1h EMA
            long_breakout = close[i] > H3[i-1]
            if vol_spike and long_breakout and close[i] > ema_1h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume spike AND price below 1h EMA
            elif vol_spike and close[i] < L3[i-1] and close[i] < ema_1h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR price crosses below 1h EMA
            exit_condition = close[i] < L3[i-1] or close[i] < ema_1h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR price crosses above 1h EMA
            exit_condition = close[i] > H3[i-1] or close[i] > ema_1h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
name = "6h_HeikinAshi_BullBearPower_1dTrend"
timeframe = "6h"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Calculate Heikin Ashi close for current bar (only needs current bar data)
    ha_close = (open_price + high + low + close) / 4
    
    # Get 1d data for Bull/Bear Power and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d EMA26 for Bull/Bear Power
    ema_26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_26_1d)
    
    # Calculate Bull Power (high - EMA26) and Bear Power (EMA26 - low)
    # Use 1d high/low for power calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_26_1d
    bear_power_1d = ema_26_1d - low_1d
    
    # Align Bull/Bear Power to 6t timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ha_close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: HA close up, Bull Power positive, price above EMA13 (uptrend)
            if (ha_close[i] > open_price[i] and 
                bull_power_aligned[i] > 0 and 
                close[i] > ema_13_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: HA close down, Bear Power positive, price below EMA13 (downtrend)
            elif (ha_close[i] < open_price[i] and 
                  bear_power_aligned[i] > 0 and 
                  close[i] < ema_13_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: HA close down OR price crosses below EMA13
            if (ha_close[i] < open_price[i] or 
                close[i] < ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: HA close up OR price crosses above EMA13
            if (ha_close[i] > open_price[i] or 
                close[i] > ema_13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
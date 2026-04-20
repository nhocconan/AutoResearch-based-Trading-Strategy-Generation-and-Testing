#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 20-period ATR on 1d for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d_arr, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR as EMA of TR (more responsive)
    atr_1d = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_20_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above EMA + volatility filter (ATR > 1% of price)
            if close_val > ema_val and atr_val > 0.01 * close_val:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA + volatility filter
            elif close_val < ema_val and atr_val > 0.01 * close_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_EMA20_1d_VolatilityFilter_Session_v1
# Uses 1d 20-period EMA for trend direction
# Requires 1d ATR > 1% of price for volatility filter (avoids low-vol chop)
# Session filter: 8-20 UTC to focus on active trading hours
# Simple crossover system with fixed 0.25 position sizing
# Designed for 4h timeframe with ~15-30 trades/year
name = "4h_EMA20_1d_VolatilityFilter_Session_v1"
timeframe = "4h"
leverage = 1.0
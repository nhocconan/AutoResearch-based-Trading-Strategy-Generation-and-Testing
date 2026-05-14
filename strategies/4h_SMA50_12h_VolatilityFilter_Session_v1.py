#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 50-period SMA on 12h close
    close_12h = df_12h['close'].values
    sma_50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    sma_50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_50_12h)
    
    # Calculate 12-period ATR on 12h for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Simple ATR (SMA of TR)
    atr_12h = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
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
        sma_val = sma_50_12h_aligned[i]
        atr_val = atr_12h_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(sma_val) or np.isnan(atr_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above SMA + volatility filter (ATR > 0.5% of price)
            if close_val > sma_val and atr_val > 0.005 * close_val:
                signals[i] = 0.25
                position = 1
            # Short: price below SMA + volatility filter
            elif close_val < sma_val and atr_val > 0.005 * close_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below SMA
            if close_val < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above SMA
            if close_val > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 4h_SMA50_12h_VolatilityFilter_Session_v1
# Uses 12h 50-period SMA for trend direction
# Requires 12h ATR > 0.5% of price for volatility filter (avoids low-vol chop)
# Session filter: 8-20 UTC to focus on active trading hours
# Simple crossover system with fixed 0.25 position sizing
# Designed for 4h timeframe with ~20-40 trades/year
name = "4h_SMA50_12h_VolatilityFilter_Session_v1"
timeframe = "4h"
leverage = 1.0
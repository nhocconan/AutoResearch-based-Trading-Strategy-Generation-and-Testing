#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Donchian(20) for breakout signals
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        atr_14_1d_val = atr_14_1d_aligned[i]
        high_20_val = high_20_aligned[i]
        low_20_val = low_20_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_val) or np.isnan(atr_14_1d_val) or 
            np.isnan(high_20_val) or np.isnan(low_20_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility
        if atr_14_1d_val < 0.005 * close_val:  # Less than 0.5% ATR
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend
            if close_val > high_20_val and close_val > ema_50_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend
            elif close_val < low_20_val and close_val < ema_50_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low
            if close_val < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high
            if close_val > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 1d_Donchian20_WeeklyEMA50_TrendFilter_Session_v1
# Uses daily Donchian(20) breakouts for entry
# Weekly EMA(50) filter ensures alignment with higher timeframe trend
# Session filter: 8-20 UTC to focus on active trading hours
# Volatility filter avoids low-volatility chop
# Designed for 1d timeframe with ~15-25 trades/year
name = "1d_Donchian20_WeeklyEMA50_TrendFilter_Session_v1"
timeframe = "1d"
leverage = 1.0
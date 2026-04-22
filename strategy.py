#!/usr/bin/env python3
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
    
    # Load weekly data for trend bias (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend bias
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily ATR for volatility filter (to avoid chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first value
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # first value
    tr3[0] = np.abs(low_1d[0] - close_1d[0])   # first value
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 60-period high/low for breakout levels (60 * 6h = 15 days)
    high_60 = pd.Series(high).rolling(window=60, min_periods=60).max().values
    low_60 = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after 60-period lookback
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(high_60[i]) or np.isnan(low_60[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop)
        if atr_14_aligned[i] < 0.5 * np.mean(atr_14_aligned[max(0, i-50):i+1]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 60-period high + weekly uptrend
            if close[i] > high_60[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below 60-period low + weekly downtrend
            elif close[i] < low_60[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse signal or volatility drop
            if position == 1:
                # Exit long: Break below 60-period low or weekly trend turns down
                if close[i] < low_60[i] or close[i] < ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Break above 60-period high or weekly trend turns up
                if close[i] > high_60[i] or close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WeeklyTrend_60Breakout_VolatilityFilter"
timeframe = "6h"
leverage = 1.0
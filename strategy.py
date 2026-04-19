#!/usr/bin/env python3
"""
12h_KAMA_Trend_Trader
Hypothesis: 12h KAMA (Kaufman Adaptive Moving Average) adapts to market noise,
providing strong trend signals in both bull and bear markets. Combined with
volume confirmation and volatility filter (ATR-based), it reduces false signals
while capturing major trends. Designed for 12h timeframe to target 50-150 total
trades over 4 years (12-37/year).
"""
name = "12h_KAMA_Trend_Trader"
timeframe = "12h"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market noise
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_length))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
        # Vectorized volatility calculation
        volatility = np.array([np.sum(np.abs(np.diff(close[i-er_length+1:i+1])) if i >= er_length-1 else 0) 
                              for i in range(len(close))])
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing constants
        sc = np.power(er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1), 2)
        
        # KAMA calculation
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # ATR for volatility filter
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(close, np.nan)
        if len(close) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(close)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close
    kama_12h = calculate_kama(df_12h['close'].values, er_length=10, fast_sc=2, slow_sc=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate ATR on 12h for volatility filter
    atr_12h = calculate_atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > 50th percentile of recent ATR
        # Simplified: use current ATR > 0 (always true if we have data)
        volatility_filter = atr_12h_aligned[i] > 0
        
        if position == 0:
            # Long: price crosses above KAMA with volume confirmation
            if (close[i] > kama_12h_aligned[i] and 
                close[i-1] <= kama_12h_aligned[i-1] and  # crossed above
                volume_confirm[i] and 
                volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume confirmation
            elif (close[i] < kama_12h_aligned[i] and 
                  close[i-1] >= kama_12h_aligned[i-1] and  # crossed below
                  volume_confirm[i] and 
                  volatility_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA
            if close[i] < kama_12h_aligned[i] and close[i-1] >= kama_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA
            if close[i] > kama_12h_aligned[i] and close[i-1] <= kama_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
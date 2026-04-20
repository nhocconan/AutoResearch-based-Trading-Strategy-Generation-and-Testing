#!/usr/bin/env python3
"""
12h_KAMA_Direction_Plus_RSI_With_Chop_Filter
Hypothesis: On 12h timeframe, use KAMA direction for trend, RSI for momentum, and Choppiness Index for regime filter.
Only trade when: KAMA indicates trend direction, RSI confirms momentum (not extreme), and market is trending (CHOP < 38.2).
In both bull and bear markets, this filters whipsaw by requiring trending regime + momentum alignment.
Target: 20-50 total trades over 4 years (5-12/year) with position size 0.25 to manage drawdown.
"""

name = "12h_KAMA_Direction_Plus_RSI_With_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_length))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
        # For array, compute rolling volatility
        volatility_arr = np.zeros_like(close)
        for i in range(len(close)):
            if i < er_length:
                volatility_arr[i] = np.nan
            else:
                volatility_arr[i] = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
        er = np.where(volatility_arr != 0, change / volatility_arr, 0)
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1d = calculate_kama(close_1d, 10, 2, 30)
    kama_1d_dir = np.where(kama_1d > np.roll(kama_1d, 1), 1, -1)
    kama_1d_dir[0] = 1  # initialize
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_dir)
    
    # Calculate 1d RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(close) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d Choppiness Index
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr_sum = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                atr_sum[i] = np.nan
            else:
                atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                hh[i] = np.nan
                ll[i] = np.nan
            else:
                hh[i] = np.max(high[i-period+1:i+1])
                ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if i < period or np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]) or hh[i] == ll[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50 and < 70, CHOP < 38.2 (trending)
            if (kama_dir_aligned[i] == 1 and 
                50 < rsi_1d_aligned[i] < 70 and 
                chop_1d_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50 and > 30, CHOP < 38.2 (trending)
            elif (kama_dir_aligned[i] == -1 and 
                  30 < rsi_1d_aligned[i] < 50 and 
                  chop_1d_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR RSI > 70 (overbought) OR CHOP > 61.8 (choppy)
            if (kama_dir_aligned[i] == -1 or 
                rsi_1d_aligned[i] > 70 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR RSI < 30 (oversold) OR CHOP > 61.8 (choppy)
            if (kama_dir_aligned[i] == 1 or 
                rsi_1d_aligned[i] < 30 or 
                chop_1d_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
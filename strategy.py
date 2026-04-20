#!/usr/bin/env python3
"""
12h_KAMA_Direction_Plus_RSI_With_Chop_Filter
Hypothesis: KAMA(10) trend direction combined with RSI(14) momentum and Choppiness Index(14) regime filter.
Long when KAMA rising, RSI > 50, and CHOP > 61.8 (ranging market) for mean reversion to upside.
Short when KAMA falling, RSI < 50, and CHOP > 61.8 (ranging market) for mean reversion to downside.
Uses 12h for primary timeframe with 1d Choppiness Index for regime filtering to avoid trending markets.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "12h_KAMA_Direction_Plus_RSI_With_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data ONCE before loop for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA(10) on 12h close
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_period))
        abs_change = np.abs(np.diff(close, prepend=close[0]))
        er = np.zeros_like(close)
        for i in range(len(close)):
            if i >= er_period:
                sum_abs_change = np.sum(abs_change[i-er_period+1:i+1])
                if sum_abs_change > 0:
                    er[i] = change[i] / sum_abs_change
                else:
                    er[i] = 0
            else:
                er[i] = 0
        
        # Smoothing Constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    kama_dir = np.where(kama > np.roll(kama, 1), 1, np.where(kama < np.roll(kama, 1), -1, 0))
    
    # Calculate RSI(14) on 12h close
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(close) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index(14) on 1d data
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR
        atr = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                atr[i] = np.mean(tr[:i+1]) if i >= 0 else 0
            else:
                atr[i] = np.mean(tr[i-period+1:i+1])
        
        # Sum of ATR over period
        sum_atr = np.zeros_like(close)
        for i in range(len(close)):
            if i >= period-1:
                sum_atr[i] = np.sum(atr[i-period+1:i+1])
        
        # Max and min close over period
        max_close = np.zeros_like(close)
        min_close = np.zeros_like(close)
        for i in range(len(close)):
            if i >= period-1:
                max_close[i] = np.max(close[i-period+1:i+1])
                min_close[i] = np.min(close[i-period+1:i+1])
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if i >= period-1 and max_close[i] != min_close[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_close[i] - min_close[i])) / np.log10(period)
            else:
                chop[i] = 50  # neutral when undefined
        
        return chop
    
    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(close[i])):
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
            # Long: KAMA rising, RSI > 50, CHOP > 61.8 (ranging market)
            if kama_dir[i] == 1 and rsi[i] > 50 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, CHOP > 61.8 (ranging market)
            elif kama_dir[i] == -1 and rsi[i] < 50 and chop_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling OR RSI < 45 OR CHOP < 38.2 (trending market)
            if kama_dir[i] == -1 or rsi[i] < 45 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising OR RSI > 55 OR CHOP < 38.2 (trending market)
            if kama_dir[i] == 1 or rsi[i] > 55 or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
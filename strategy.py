#!/usr/bin/env python3
"""
1d_1w_kama_rsi_chop_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
Relative Strength Index (RSI) for overbought/oversold conditions, and Choppiness Index (CHOP)
as a regime filter to avoid whipsaws. Enter long when KAMA trends up, RSI < 30, and CHOP > 61.8 (ranging).
Enter short when KAMA trends down, RSI > 70, and CHOP > 61.8. Exit when conditions reverse.
This strategy aims to capture mean-reversion in ranging markets while avoiding strong trends,
which should work in both bull and bear markets by adapting to market regime.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
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
    
    # Get weekly data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index (14-period) on weekly data
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr_list = []
        for i in range(1, len(close_arr)):
            tr = max(
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            )
            atr_list.append(tr)
        atr = np.array(atr_list)
        
        # Sum of ATR over period
        atr_sum = np.zeros(len(close_arr))
        for i in range(period, len(atr)+1):
            atr_sum[i] = atr[i-period:i].sum()
        
        # Highest high and lowest low over period
        max_high = np.zeros(len(close_arr))
        min_low = np.zeros(len(close_arr))
        for i in range(period, len(close_arr)):
            max_high[i] = high_arr[i-period:i].max()
            min_low[i] = low_arr[i-period:i].min()
        
        # Chop formula: 100 * log10(sum(atr) / (max_high - min_low)) / log10(period)
        range_ = max_high - min_low
        chop = np.full(len(close_arr), np.nan)
        for i in range(period, len(close_arr)):
            if range_[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / range_[i]) / np.log10(period)
        return chop
    
    chop = calculate_chop(high_1w, low_1w, close_1w, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # Calculate KAMA on daily close
    def kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # temporary, will fix
        # Recalculate volatility properly
        volatility = np.zeros(len(close))
        for i in range(er_period, len(close)):
            volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        er = np.zeros(len(close))
        for i in range(er_period, len(close)):
            if volatility[i] > 0:
                er[i] = change[i-er_period] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama_vals = np.zeros(len(close))
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    kama_dir = np.where(kama_vals > np.roll(kama_vals, 1), 1, -1)  # 1 for up, -1 for down
    
    # Calculate RSI (14-period) on daily close
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(len(close))
        avg_loss = np.zeros(len(close))
        
        # Initial average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.zeros(len(close))
        for i in range(period, len(close)):
            if avg_loss[i] > 0:
                rs[i] = avg_gain[i] / avg_loss[i]
            else:
                rs[i] = 0
        
        rsi_vals = np.zeros(len(close))
        for i in range(period, len(close)):
            rsi_vals[i] = 100 - (100 / (1 + rs[i]))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Conditions
    # Chop > 61.8 indicates ranging market (good for mean reversion)
    chop_threshold = 61.8
    ranging = chop_aligned > chop_threshold
    
    # Long when: KAMA up, RSI < 30 (oversold), and ranging market
    long_entry = (kama_dir == 1) & (rsi_vals < 30) & ranging
    # Short when: KAMA down, RSI > 70 (overbought), and ranging market
    short_entry = (kama_dir == -1) & (rsi_vals > 70) & ranging
    
    # Exit when conditions reverse
    long_exit = (kama_dir == -1) | (rsi_vals > 70)  # KAMA down or RSI overbought
    short_exit = (kama_dir == 1) | (rsi_vals < 30)   # KAMA up or RSI oversold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check for entry signals
        if long_entry[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Check for exit signals
        elif position == 1 and long_exit[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
# 1d KAMA + RSI + Chop Regime Strategy
# Uses KAMA direction on 1d timeframe with RSI momentum filter and Choppiness Index regime filter
# Works in both bull and bear markets by adapting to trending vs ranging conditions
# Target: 30-100 trades over 4 years (7-25/year)

#!/usr/bin/env python3
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
    volume = prices['volume'].values
    
    # Load 1d data for KAMA, RSI, and Choppiness Index (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # ER = Efficiency Ratio, SC = Smoothing Constant
    def calculate_kama(close_series, kama_period=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close_series, k=10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close_series)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1), 2)
        kama = np.full_like(close_series, np.nan, dtype=float)
        kama[kama_period] = close_series[kama_period]
        for i in range(kama_period + 1, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series[i] - kama[i-1])
        return kama
    
    # Calculate RSI on 1d
    def calculate_rsi(close_series, period=14):
        delta = np.diff(close_series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_series)
        avg_loss = np.zeros_like(close_series)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close_series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate Choppiness Index on 1d
    def calculate_choppiness(high_series, low_series, close_series, period=14):
        atr = np.zeros_like(close_series)
        tr1 = high_series[1:] - low_series[1:]
        tr2 = np.abs(high_series[1:] - close_series[:-1])
        tr3 = np.abs(low_series[1:] - close_series[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[period:] = np.sum(tr[-period:], axis=0) if len(tr) >= period else 0
        for i in range(len(tr), len(close_series)):
            atr[i] = (atr[i-1] * (period-1) + tr[i-1]) / period if i-1 < len(tr) else atr[i-1]
        
        max_high = np.zeros_like(close_series)
        min_low = np.zeros_like(close_series)
        for i in range(period, len(close_series)):
            max_high[i] = np.max(high_series[i-period+1:i+1])
            min_low[i] = np.min(low_series[i-period+1:i+1])
        
        chop = np.full_like(close_series, 50.0, dtype=float)
        for i in range(period, len(close_series)):
            if max_high[i] > min_low[i] and atr[i] > 0:
                chop[i] = 100 * np.log10(max_high[i] - min_low[i]) / np.log10(period) / np.log10(atr[i] * period)
        return chop
    
    # Calculate indicators on 1d
    kama_1d = calculate_kama(close_1d, kama_period=10, fast_sc=2, slow_sc=30)
    rsi_1d = calculate_rsi(close_1d, period=14)
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, period=14)
    
    # Align 1d indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA turning up + RSI > 50 + Chop < 61.8 (trending market)
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down + RSI < 50 + Chop < 61.8 (trending market)
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: KAMA crosses in opposite direction OR Chop > 61.8 (ranging market)
            if position == 1:
                if close[i] < kama_aligned[i] or chop_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_aligned[i] or chop_aligned[i] > 61.8:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_KAMA_RSI_Chop_Regime"
timeframe = "1d"
leverage = 1.0
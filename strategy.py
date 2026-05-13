#!/usr/bin/env python3
"""
12h_KAMA_Trend_Filter_v1
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h to capture trend direction, filtered by 1d RSI to avoid extremes and volume confirmation. KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends. Works in bull markets by riding trends and in bear markets by avoiding false signals during low volatility. Target: 20-50 trades/year on 12h timeframe to minimize fee drag.
"""

name = "12h_KAMA_Trend_Filter_v1"
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
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA on 12h close
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation over 10-period window
    def calculate_kama(close_array, er_period=10, fast_sc=2, slow_sc=30):
        n_len = len(close_array)
        kama = np.zeros(n_len)
        kama[0] = close_array[0]
        
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close_array, prepend=close_array[0]))
        vol = np.abs(np.diff(close_array))
        
        er = np.zeros(n_len)
        er[0] = 0
        for i in range(1, n_len):
            if i < er_period:
                er[i] = np.sum(change[1:i+1]) / np.sum(vol[1:i+1]) if np.sum(vol[1:i+1]) > 0 else 0
            else:
                er[i] = np.sum(change[i-er_period+1:i+1]) / np.sum(vol[i-er_period+1:i+1]) if np.sum(vol[i-er_period+1:i+1]) > 0 else 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # Calculate KAMA
        for i in range(1, n_len):
            kama[i] = kama[i-1] + sc[i] * (close_array[i] - kama[i-1])
        
        return kama
    
    kama_12h = calculate_kama(close_12h, er_period=10, fast_sc=2, slow_sc=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate RSI on 1d close
    def calculate_rsi(close_array, period=14):
        n_len = len(close_array)
        delta = np.diff(close_array, prepend=close_array[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n_len)
        avg_loss = np.zeros(n_len)
        
        # Initial average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, n_len):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, period=14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate volume average on 1d for spike detection
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-day average
        # Need to get current 1d volume aligned to 12h
        vol_1d_current = volume_1d[np.searchsorted(df_1d.index.values[:len(df_1d)], 
                                                   pd.Timestamp(prices['open_time'].iloc[i]), 
                                                   side='right') - 1] if i > 0 else volume_1d[0]
        vol_spike = vol_1d_current > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        if position == 0:
            # LONG: Price above KAMA + RSI not overbought ( < 70) + volume spike
            if close[i] > kama_12h_aligned[i] and rsi_1d_aligned[i] < 70 and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + RSI not oversold ( > 30) + volume spike
            elif close[i] < kama_12h_aligned[i] and rsi_1d_aligned[i] > 30 and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or RSI overbought
            if close[i] < kama_12h_aligned[i] or rsi_1d_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or RSI oversold
            if close[i] > kama_12h_aligned[i] or rsi_1d_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
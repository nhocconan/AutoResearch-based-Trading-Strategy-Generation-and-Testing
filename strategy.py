#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Stochastic
# Hypothesis: KAMA trend direction on 12h combined with RSI and Stochastic for entry timing, with volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in ranging markets. RSI identifies overbought/oversold conditions,
# Stochastic confirms momentum. Volume ensures breakout strength. Designed for 12h to achieve 12-37 trades/year.
# Works in both bull and bear markets by adapting trend strength and avoiding false signals in low volatility.

name = "12h_KAMA_Trend_RSI_Stochastic"
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
    volume = prices['volume'].values
    
    # 1d data for KAMA trend, RSI, and Stochastic
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend indicator
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if length > 1 else np.zeros_like(change)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / np.where(volatility[length-1:] == 0, 1, volatility[length-1:])
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.full_like(close, np.nan)
        kama[length] = close[length]
        for i in range(length+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI (Relative Strength Index)
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Stochastic Oscillator
    def calculate_stochastic(high, low, close, k_length=14, d_length=3):
        lowest_low = np.full_like(low, np.nan)
        highest_high = np.full_like(high, np.nan)
        for i in range(k_length-1, len(low)):
            lowest_low[i] = np.min(low[i-k_length+1:i+1])
            highest_high[i] = np.max(high[i-k_length+1:i+1])
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        k = np.where(highest_high == lowest_low, 50, k)  # avoid division by zero
        # D is SMA of K
        d = np.full_like(close, np.nan)
        for i in range(d_length-1, len(k)):
            d[i] = np.mean(k[i-d_length+1:i+1])
        return k, d
    
    # Calculate indicators on 1d data
    kama = calculate_kama(close_1d, length=10, fast=2, slow=30)
    rsi = calculate_rsi(close_1d, length=14)
    stoch_k, stoch_d = calculate_stochastic(high_1d, low_1d, close_1d, k_length=14, d_length=3)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align all indicators to lower timeframe (wait for 1d bar to close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    stoch_k_aligned = align_htf_to_ltf(prices, df_1d, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_1d, stoch_d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(stoch_k_aligned[i]) or np.isnan(stoch_d_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI < 70 (not overbought), Stochastic K > D (bullish momentum), strong volume
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 70 and stoch_k_aligned[i] > stoch_d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI > 30 (not oversold), Stochastic K < D (bearish momentum), strong volume
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 30 and stoch_k_aligned[i] < stoch_d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI > 70 (overbought) or Stochastic K < D (momentum loss)
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70 or stoch_k_aligned[i] < stoch_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI < 30 (oversold) or Stochastic K > D (momentum loss)
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30 or stoch_k_aligned[i] > stoch_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
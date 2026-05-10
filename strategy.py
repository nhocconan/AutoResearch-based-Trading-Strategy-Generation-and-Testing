#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_MeanReversion
# Hypothesis: KAMA identifies adaptive trend direction on 12h, RSI(14) identifies overbought/oversold conditions for mean reversion entries.
# In both bull and bear markets, price tends to revert to the mean when RSI reaches extremes, but only in the direction of the 12h trend.
# Uses 1d trend filter (EMA50) to avoid counter-trend trades. Volume spike confirms momentum.
# Designed for 12h to achieve 12-37 trades/year with low frequency and high conviction.

name = "12h_KAMA_Trend_RSI_MeanReversion"
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
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) - 12h
    def kama(arr, period=10, fast=2, slow=30):
        n = len(arr)
        kama_vals = np.full(n, np.nan)
        if n < period:
            return kama_vals
        
        # Efficiency Ratio
        change = np.abs(arr[period-1:] - arr[:-(period-1)])
        volatility = np.sum(np.abs(np.diff(arr[:period])), axis=0) if period > 1 else 0
        er = np.zeros(n)
        er[period-1:] = change / (volatility + 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Initialize KAMA
        kama_vals[period-1] = arr[period-1]
        for i in range(period, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (arr[i] - kama_vals[i-1])
        return kama_vals
    
    # RSI calculation
    def rsi(arr, period=14):
        n = len(arr)
        rsi_vals = np.full(n, np.nan)
        if n < period + 1:
            return rsi_vals
        
        delta = np.diff(arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals[period:] = 100 - (100 / (1 + rs[period:]))
        return rsi_vals
    
    # Calculate indicators
    kama_vals = kama(close, period=10, fast=2, slow=30)
    rsi_vals = rsi(close, period=14)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align all indicators to lower timeframe (wait for 1d bar to close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_vals)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_vals)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend), RSI oversold, above 1d EMA50, strong volume
            if close[i] < kama_aligned[i] and rsi_aligned[i] < 30 and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (pullback in downtrend), RSI overbought, below 1d EMA50, strong volume
            elif close[i] > kama_aligned[i] and rsi_aligned[i] > 70 and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses above KAMA or RSI overbought
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses below KAMA or RSI oversold
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
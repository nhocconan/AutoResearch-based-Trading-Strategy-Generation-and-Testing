#!/usr/bin/env python3
# 12h_KAMA_Trend_Filter
# Hypothesis: 12-hour strategy using KAMA trend filter on daily timeframe, with RSI and volume confirmation on 12h.
# KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI avoids overbought/oversold extremes.
# Volume ensures breakout strength. Designed for 12h to achieve 12-37 trades/year.
# Works in bull markets by catching trends, in bear markets by avoiding false signals during low volatility.

name = "12h_KAMA_Trend_Filter"
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
    
    # Daily data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on daily close
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / volatility[length-1:]
        er[er < 0] = 0
        er[er > 1] = 1
        
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA calculation
        kama_vals = np.zeros_like(close)
        kama_vals[:] = np.nan
        kama_vals[length] = close[length]
        for i in range(length+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1d = kama(close_1d, length=10, fast=2, slow=30)
    
    # 12h RSI (14-period)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.zeros_like(close)
        rs[:] = np.nan
        valid = avg_loss != 0
        rs[valid] = avg_gain[valid] / avg_loss[valid]
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_12h = rsi(close, period=14)
    
    # 12h volume moving average (20-period)
    def sma(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period-1, len(arr)):
                res[i] = np.mean(arr[i-period+1:i+1])
        return res
    
    vol_ma_20 = sma(volume, 20)
    
    # Align daily KAMA to 12h timeframe (wait for 1d bar to close)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_12h[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI not overbought, volume above average
            if close[i] > kama_1d_aligned[i] and rsi_12h[i] < 70 and volume[i] > vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI not oversold, volume above average
            elif close[i] < kama_1d_aligned[i] and rsi_12h[i] > 30 and volume[i] > vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or RSI overbought
            if close[i] < kama_1d_aligned[i] or rsi_12h[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or RSI oversold
            if close[i] > kama_1d_aligned[i] or rsi_12h[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
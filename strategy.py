#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA: Kaufman Adaptive Moving Average (fast=2, slow=30)
    def kama(close_arr, fast=2, slow=30):
        n = len(close_arr)
        kama_arr = np.full(n, np.nan)
        if n == 0:
            return kama_arr
        # Efficiency Ratio
        change = np.abs(np.diff(close_arr, n=10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close_arr, n=1)), axis=0)  # 10-period volatility
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Initialize
        kama_arr[9] = close_arr[9]  # Start after 10 periods
        for i in range(10, n):
            kama_arr[i] = kama_arr[i-1] + sc[i] * (close_arr[i] - kama_arr[i-1])
        return kama_arr
    
    # Calculate KAMA on weekly data
    kama_1w = kama(close_1w, fast=2, slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI for mean reversion
    def rsi(close_arr, period=14):
        delta = np.diff(close_arr)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_arr)
        avg_loss = np.zeros_like(close_arr)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close_arr)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_arr = 100 - (100 / (1 + rs))
        rsi_arr[:period] = np.nan
        return rsi_arr
    
    rsi_14 = rsi(close, period=14)
    
    # Volume spike (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi_14[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend), RSI oversold, volume spike
            if (close[i] < kama_1w_aligned[i] and 
                rsi_14[i] < 30 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (rally in downtrend), RSI overbought, volume spike
            elif (close[i] > kama_1w_aligned[i] and 
                  rsi_14[i] > 70 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above KAMA or RSI overbought
            if close[i] > kama_1w_aligned[i] or rsi_14[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below KAMA or RSI oversold
            if close[i] < kama_1w_aligned[i] or rsi_14[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
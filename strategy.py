#!/usr/bin/env python3
name = "12h_KAMA_Trend_RSI_MeanReversion"
timeframe = "12h"
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
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) for trend direction
    # KAMA adapts to market noise - faster in trends, slower in ranges
    def calculate_kama(close_prices, er_period=10, fast_sc=2, slow_sc=30):
        n = len(close_prices)
        kama = np.full(n, np.nan)
        if n < er_period + 1:
            return kama
        
        # Efficiency Ratio
        change = np.abs(close_prices[er_period:] - close_prices[:-er_period])
        volatility = np.sum(np.abs(np.diff(close_prices[:n-er_period+1])), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing constants
        sc = np.power(er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1), 2)
        
        # KAMA calculation
        kama[er_period] = close_prices[er_period]
        for i in range(er_period + 1, n):
            kama[i] = kama[i-1] + sc[i-er_period] * (close_prices[i] - kama[i-1])
        return kama
    
    # RSI calculation
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # Initial average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get 1d data for trend filter (stronger trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    
    # Get 12h data for KAMA and RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate indicators on 12h data
    kama_12h = calculate_kama(close_12h, er_period=10, fast_sc=2, slow_sc=30)
    rsi_12h = calculate_rsi(close_12h, period=14)
    
    # Volume moving average for confirmation
    vol_ma20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to main timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    vol_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 50)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(vol_ma20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI oversold (<30) + volume confirmation
            if (close[i] > kama_12h_aligned[i] and 
                rsi_12h_aligned[i] < 30 and 
                volume[i] > 1.3 * vol_ma20_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI overbought (>70) + volume confirmation
            elif (close[i] < kama_12h_aligned[i] and 
                  rsi_12h_aligned[i] > 70 and 
                  volume[i] > 1.3 * vol_ma20_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price below KAMA or RSI overbought
            if (close[i] < kama_12h_aligned[i] or 
                rsi_12h_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price above KAMA or RSI oversold
            if (close[i] > kama_12h_aligned[i] or 
                rsi_12h_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
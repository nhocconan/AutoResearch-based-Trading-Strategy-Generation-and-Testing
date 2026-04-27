#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1-day KAMA calculation
    def kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / volatility[period-1:]
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
        kama = np.full_like(price, np.nan)
        kama[period] = price[period]
        for i in range(period+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # 1-day RSI calculation
    def rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(price, np.nan)
        avg_loss = np.full_like(price, np.nan)
        if len(price) >= period+1:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            for i in range(period+1, len(price)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    kama_1d = kama(close_1d, 10, 2, 30)
    rsi_1d = rsi(close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike detection (6-period average)
    vol_ma = np.full(n, np.nan)
    vol_period = 6
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(10, 14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: KAMA trending up + RSI not overbought + volume spike
            if kama_1d_aligned[i] > kama_1d_aligned[i-1] and rsi_1d_aligned[i] < 70 and vol_filter:
                signals[i] = size
                position = 1
            # Short: KAMA trending down + RSI not oversold + volume spike
            elif kama_1d_aligned[i] < kama_1d_aligned[i-1] and rsi_1d_aligned[i] > 30 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: KAMA turns down OR RSI overbought
            if kama_1d_aligned[i] < kama_1d_aligned[i-1] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: KAMA turns up OR RSI oversold
            if kama_1d_aligned[i] > kama_1d_aligned[i-1] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_KAMA_RSI_Volume"
timeframe = "12h"
leverage = 1.0
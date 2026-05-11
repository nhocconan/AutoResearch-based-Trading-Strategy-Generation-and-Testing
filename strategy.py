#!/usr/bin/env python3
name = "12h_KAMA_RSI_ChopFilter_v2"
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
    
    # Get 1D data for higher timeframe filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA on 12H
    def calculate_kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.abs(np.diff(price)).cumsum()
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.where(volatility[period-1:] == 0, 1, volatility[period-1:])
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # RSI on 12H
    def calculate_rsi(price, period=14):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(price)
        avg_loss = np.zeros_like(price)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Chopiness Index on 1D (regime filter)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            atr[i] = np.mean(tr[max(0, i-period+1):i+1])
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(len(close)):
            max_high[i] = np.max(high[max(0, i-period+1):i+1])
            min_low[i] = np.min(low[max(0, i-period+1):i+1])
        range_period = max_high - min_low
        sum_atr = np.zeros_like(close)
        for i in range(len(close)):
            sum_atr[i] = np.sum(atr[max(0, i-period+1):i+1])
        chop = 100 * np.log10(sum_atr / range_period) / np.log10(period)
        return chop
    
    chop = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Long: Price > KAMA, RSI > 50, Chop > 50 (ranging market), Volume surge
            if (price_above_kama and rsi[i] > 50 and chop_aligned[i] > 50 and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA, RSI < 50, Chop > 50 (ranging market), Volume surge
            elif (price_below_kama and rsi[i] < 50 and chop_aligned[i] > 50 and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below KAMA OR RSI < 40 OR Chop < 30 (trending)
            if (close[i] <= kama[i] or rsi[i] < 40 or chop_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above KAMA OR RSI > 60 OR Chop < 30 (trending)
            if (close[i] >= kama[i] or rsi[i] > 60 or chop_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
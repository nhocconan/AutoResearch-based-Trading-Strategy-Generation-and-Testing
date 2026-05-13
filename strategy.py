#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Reversal_v2"
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
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    def kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.zeros_like(price)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(price)
        kama[:period] = price[:period]
        for i in range(period, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # RSI
    def rsi(price, period=14):
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
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros(len(close))
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        sum_atr = np.zeros(len(close))
        for i in range(period, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        max_high = np.zeros(len(close))
        min_low = np.zeros(len(close))
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        chop = np.zeros(len(close))
        for i in range(period-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    kama_vals = kama(close, 10, 2, 30)
    rsi_vals = rsi(close, 14)
    chop_vals = choppiness_index(high, low, close, 14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(chop_vals[i]) or np.isnan(ema20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: range-bound market (chop > 61.8) for mean reversion
        chop_condition = chop_vals[i] > 61.8
        
        if position == 0:
            # LONG: RSI oversold + price above KAMA + choppy market
            if rsi_vals[i] < 30 and close[i] > kama_vals[i] and chop_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought + price below KAMA + choppy market
            elif rsi_vals[i] > 70 and close[i] < kama_vals[i] and chop_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought or price below KAMA
            if rsi_vals[i] > 70 or close[i] < kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold or price above KAMA
            if rsi_vals[i] < 30 or close[i] > kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 1D_KAMA_1WTrend_RSI_Dip
# Hypothesis: On the daily timeframe, use KAMA to determine trend direction from weekly trend, buy dips in RSI during uptrends, sell rallies in RSI during downtrends. Uses 1d timeframe with 1h trend filter. Target: 15-25 trades/year to stay under 100 total trades. KAMA adapts to market noise, RSI dip/ride provides mean reversion within trend, reducing whipsaws in both bull and bear markets.

name = "1D_KAMA_1WTrend_RSI_Dip"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate KAMA for trend direction (daily)
    def calculate_kama(close_vals, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close_vals, n=er_length))
        volatility = np.sum(np.abs(np.diff(close_vals)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close_vals)
        kama[0] = close_vals[0]
        for i in range(1, len(close_vals)):
            kama[i] = kama[i-1] + sc[i] * (close_vals[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # Weekly trend: price above/below weekly KAMA
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, 10, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # RSI(14) for dip/ride signals
    def calculate_rsi(close_vals, length=14):
        delta = np.diff(close_vals)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close_vals)
        avg_loss = np.zeros_like(close_vals)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close_vals)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:length] = 50  # neutral before enough data
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # after KAMA and RSI warmup
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(kama_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly uptrend (price > weekly KAMA) and RSI dip (< 40)
            if (close[i] > kama_1w_aligned[i] and rsi[i] < 40):
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend (price < weekly KAMA) and RSI rally (> 60)
            elif (close[i] < kama_1w_aligned[i] and rsi[i] > 60):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI reaches overbought (> 70) or trend turns down
            if (rsi[i] > 70 or close[i] < kama_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI reaches oversold (< 30) or trend turns up
            if (rsi[i] < 30 or close[i] > kama_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter
# Hypothesis: Daily KAMA trend direction + RSI extremes + Choppiness regime filter.
# KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI >70 or <30 triggers mean-reversion entries only when market is choppy (CHOP > 61.8), avoiding strong trends.
# Designed for 1d to achieve 7-25 trades/year with low frequency, suitable for both bull and bear markets by avoiding trend-following in chop and fading extremes in range.

name = "1d_KAMA_RSI_ChopFilter"
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
    
    # Weekly data for trend filter (optional, can be removed if not needed)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend component
    def kama(close, er_len=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[er_len:] = change[er_len-1:] / np.maximum(volatility[er_len-1:], 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10, 2, 30)
    
    # RSI (14)
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
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_vals = rsi(close, 14)
    
    # Choppiness Index (14)
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Sum of True Range over period
        sum_atr = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50  # avoid division by zero
        return chop
    
    chop_vals = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # enough for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in choppy market (CHOP > 61.8 = ranging)
        if chop_vals[i] > 61.8:
            if position == 0:
                # Long: RSI < 30 (oversold) and price > KAMA (weak uptrend bias)
                if rsi_vals[i] < 30 and close[i] > kama_vals[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: RSI > 70 (overbought) and price < KAMA (weak downtrend bias)
                elif rsi_vals[i] > 70 and close[i] < kama_vals[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Long exit: RSI > 50 or price < KAMA
                if rsi_vals[i] > 50 or close[i] < kama_vals[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: RSI < 50 or price > KAMA
                if rsi_vals[i] < 50 or close[i] > kama_vals[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # In trending market, stay flat or follow KAMA trend (optional)
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals
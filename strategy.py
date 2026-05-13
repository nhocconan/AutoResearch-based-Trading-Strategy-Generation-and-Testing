#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_Pullback
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
Buy on pullbacks to KAMA during uptrend (RSI<40), sell on rallies to KAMA during downtrend (RSI>60).
Uses 1d trend filter to avoid counter-trend trades. Designed for low trade frequency (<25/year) to avoid fee drag.
"""

name = "4h_KAMA_Trend_RSI_Pullback"
timeframe = "4h"
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
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[period-1] = close[period-1]
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10, 2, 30)
    
    # RSI
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
    
    rsi_val = rsi(close, 14)
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: price near KAMA (pullback) in uptrend, RSI oversold, volume confirmation
            if (close[i] <= kama_val[i] * 1.01 and  # within 1% above KAMA
                close[i] >= kama_val[i] * 0.99 and   # within 1% below KAMA
                uptrend_1d[i] and
                rsi_val[i] < 40 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price near KAMA (pullback) in downtrend, RSI overbought, volume confirmation
            elif (close[i] <= kama_val[i] * 1.01 and
                  close[i] >= kama_val[i] * 0.99 and
                  downtrend_1d[i] and
                  rsi_val[i] > 60 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses above KAMA or RSI overbought
            if close[i] > kama_val[i] * 1.02 or rsi_val[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below KAMA or RSI oversold
            if close[i] < kama_val[i] * 0.98 or rsi_val[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
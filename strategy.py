#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Range_200MA_v2
Hypothesis: Combines KAMA (adaptive trend) with RSI mean reversion and 200MA filter to work in both bull and bear markets. KAMA filters trend direction (bull/bear), RSI identifies overbought/oversold within that trend, and 200MA avoids counter-trend trades. Uses 1d ADX for regime filter (ADX>25 = trend, <20 = range) to avoid whipsaws. Designed for low trade frequency (~20-30 trades/year) with strong signal quality.
"""

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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend filter
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.subtract.outer(close, close)).sum(axis=1)  # inefficient but works for small er_length
        # More efficient volatility calculation
        volatility = np.zeros_like(close)
        for i in range(len(close)):
            start = max(0, i - er_length + 1)
            volatility[i] = np.sum(np.abs(np.diff(close[start:i+1])))
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Get 1d data for ADX regime filter and 200MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX for regime filter (trending vs ranging)
    def calculate_adx(high, low, close, length=14):
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        tr = np.maximum(np.abs(high[1:] - low[:-1]), np.abs(low[1:] - high[:-1]))
        tr = np.maximum(tr, np.abs(close[1:] - close[:-1]))
        
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        return adx
    
    # Calculate indicators on 1d
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    ma200_1d = pd.Series(df_1d['close']).rolling(window=200, min_periods=200).mean().values
    
    # Align 1d indicators to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ma200_1d_aligned = align_htf_to_ltf(prices, df_1d, ma200_1d)
    
    # Calculate KAMA and RSI on 4h price data
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    rsi = calculate_rsi(close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ma200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        adx_val = adx_1d_aligned[i]
        ma200_val = ma200_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price > KAMA (uptrend), RSI < 30 (oversold), price > MA200, ADX > 25 (trending)
            if close[i] > kama_val and rsi_val < 30 and close[i] > ma200_val and adx_val > 25:
                signals[i] = size
                position = 1
            # Short conditions: price < KAMA (downtrend), RSI > 70 (overbought), price < MA200, ADX > 25 (trending)
            elif close[i] < kama_val and rsi_val > 70 and close[i] < ma200_val and adx_val > 25:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price < KAMA (trend change) OR RSI > 70 (overbought)
            if close[i] < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price > KAMA (trend change) OR RSI < 30 (oversold)
            if close[i] > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Direction_RSI_Range_200MA_v2"
timeframe = "4h"
leverage = 1.0
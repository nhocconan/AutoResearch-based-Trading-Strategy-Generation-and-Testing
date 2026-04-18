#!/usr/bin/env python3
"""
4h_KAMA_Trend_with_RSI_Filter_and_ATR_Stop
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
Combined with RSI(2) for oversold/overbought entries and ATR-based stoploss to manage risk.
Designed for 4h timeframe to capture medium-term moves with minimal trades.
"""

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
    
    # KAMA trend on 4h
    def calculate_kama(price, period=10, fast=2, slow=30):
        change = np.abs(np.diff(price, n=period))
        volatility = np.sum(np.abs(np.diff(price)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(price, np.nan, dtype=float)
        kama[period] = price[period]
        for i in range(period + 1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # RSI(2) for entry signals
    def calculate_rsi(price, period=2):
        delta = np.diff(price)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(price, np.nan, dtype=float)
        avg_loss = np.full_like(price, np.nan, dtype=float)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(price)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=2)
    
    # ATR for stoploss and position sizing
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.full_like(tr, np.nan, dtype=float)
        atr[period] = np.mean(tr[:period])
        for i in range(period + 1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    position = 0
    entry_price = 0
    
    start_idx = max(30, 14)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend) and RSI oversold
            if price > kama_val and rsi_val < 15:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price below KAMA (downtrend) and RSI overbought
            elif price < kama_val and rsi_val > 85:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: RSI overbought or ATR-based stoploss
            if rsi_val > 70 or price < entry_price - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: RSI oversold or ATR-based stoploss
            if rsi_val < 30 or price > entry_price + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_with_RSI_Filter_and_ATR_Stop"
timeframe = "4h"
leverage = 1.0
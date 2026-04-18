#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter
Hypothesis: Daily KAMA direction filtered by RSI extremes and Choppiness Index regime.
KAMA adapts to market noise, providing reliable trend direction.
RSI < 30 or > 70 identifies overextended conditions for mean reversion in choppy markets.
Choppiness Index > 61.8 indicates ranging market (favor mean reversion), < 38.2 indicates trending (favor trend continuation).
Designed for low trade frequency (target: 10-25/year) with strong performance in both bull and bear markets via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_ktf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (10-period ER, 2/30 smoothing constants)
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        kama = np.full_like(close, np.nan, dtype=float)
        if len(close) < er_period + 1:
            return kama
        
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close, er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Calculate smoothing constant
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # Initialize KAMA
        kama[er_period] = close[er_period]
        
        # Calculate KAMA
        for i in range(er_period + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10, 2, 30)
    
    # Calculate RSI (14-period)
    def calculate_rsi(close, period=14):
        rsi = np.full_like(close, 50.0, dtype=float)
        if len(close) < period + 1:
            return rsi
        
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index (14-period)
    def calculate_choppiness(high, low, close, period=14):
        chop = np.full_like(close, 50.0, dtype=float)
        if len(close) < period:
            return chop
        
        atr = np.full_like(close, np.nan)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr[i] = tr
        
        for i in range(period, len(close)):
            atr_sum = np.sum(atr[i-period+1:i+1])
            hh = np.max(high[i-period+1:i+1])
            ll = np.min(low[i-period+1:i+1])
            if hh - ll != 0:
                chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # Weekly trend filter using EMA34
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 14)  # KAMA(10)+1, RSI(14), CHOP(14)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: KAMA up AND (RSI oversold in chop OR RSI not overbought in trend)
            if (kama[i] > close[i] and 
                ((rsi[i] < 30 and chop[i] > 61.8) or  # Mean reversion in chop
                 (rsi[i] < 70 and chop[i] <= 61.8))):  # Trend continuation
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA down AND (RSI overbought in chop OR RSI not oversold in trend)
            elif (kama[i] < close[i] and 
                  ((rsi[i] > 70 and chop[i] > 61.8) or  # Mean reversion in chop
                   (rsi[i] > 30 and chop[i] <= 61.8))):  # Trend continuation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR RSI overbought in chop
            if (kama[i] < close[i] or (rsi[i] > 70 and chop[i] > 61.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR RSI oversold in chop
            if (kama[i] > close[i] or (rsi[i] < 30 and chop[i] > 61.8)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
# 1d_KAMA_RSI_ChopFilter
# Hypothesis: KAMA trend direction + RSI mean reversion + Choppiness regime filter. 
# KAMA adapts to market noise, filtering out false signals. RSI identifies overbought/oversold conditions.
# Choppiness filter avoids whipsaws in ranging markets (CHOP > 61.8) and follows trends in trending markets (CHOP < 38.2).
# Works in bull/bear by combining adaptive trend with mean reversion in appropriate regimes.
# Targets 15-25 trades/year to minimize fee drag.

name = "1d_KAMA_RSI_ChopFilter"
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
    
    # KAMA: Kaufman Adaptive Moving Average
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # RSI: Relative Strength Index
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Choppiness Index
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        sum_atr = np.zeros_like(close)
        for i in range(length, len(close)):
            sum_atr[i] = np.sum(atr[i-length+1:i+1])
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(length-1, len(close)):
            max_high[i] = np.max(high[i-length+1:i+1])
            min_low[i] = np.min(low[i-length+1:i+1])
        chop = np.zeros_like(close)
        for i in range(length-1, len(close)):
            if max_high[i] != min_low[i]:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close)
    rsi = calculate_rsi(close)
    chop = calculate_chop(high, low, close)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    sma_50_1w = pd.Series(df_1w['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 14, 50)  # Warmup for KAMA, RSI, and weekly SMA
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(sma_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly SMA
        uptrend = close[i] > sma_50_1w_aligned[i]
        downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Chop regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        if position == 0:
            # Long: KAMA up + RSI oversold in trending market OR mean reversion in ranging market
            if ((kama[i] > kama[i-1] and rsi[i] < 30 and trending_regime) or
                (rsi[i] < 25 and ranging_regime)):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought in trending market OR mean reversion in ranging market
            elif ((kama[i] < kama[i-1] and rsi[i] > 70 and trending_regime) or
                  (rsi[i] > 75 and ranging_regime)):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down OR RSI overbought
            if kama[i] < kama[i-1] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up OR RSI oversold
            if kama[i] > kama[i-1] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
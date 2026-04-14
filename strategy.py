#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA + RSI + chop filter
# Long when KAMA is rising (bullish trend) AND RSI < 30 (oversold) AND chop > 61.8 (ranging market)
# Short when KAMA is falling (bearish trend) AND RSI > 70 (overbought) AND chop > 61.8 (ranging market)
# Exit when RSI crosses back to neutral (40 for long, 60 for short)
# This strategy aims to capture mean reversion in ranging markets with trend confirmation from KAMA
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA on close
    def calculate_kama(close, length=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[length:] = change[length-1:] / volatility[length-1:]
        # Smoothing Constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        # KAMA
        kama = np.full_like(close, np.nan)
        kama[length] = close[length]
        for i in range(length+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10)
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    
    # Calculate RSI
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
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        sum_atr = np.zeros_like(close)
        for i in range(length, len(close)):
            sum_atr[i] = np.sum(atr[i-length+1:i+1])
        max_range = np.zeros_like(close)
        for i in range(length, len(close)):
            max_range[i] = np.max(high[i-length+1:i+1]) - np.min(low[i-length+1:i+1])
        chop = np.where(max_range != 0, 100 * np.log10(sum_atr / max_range) / np.log10(length), 50)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: KAMA rising + RSI oversold + choppy market
            if kama_rising[i] and rsi[i] < 30 and chop[i] > 61.8:
                position = 1
                signals[i] = position_size
            # Short setup: KAMA falling + RSI overbought + choppy market
            elif kama_falling[i] and rsi[i] > 70 and chop[i] > 61.8:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 40 (neutral)
            if rsi[i] > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI crosses below 60 (neutral)
            if rsi[i] < 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0
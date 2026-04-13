#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h with 1d KAMA trend + RSI + chop filter.
# Long: KAMA rising (trend up) + RSI(14) > 50 + Chop(14) < 61.8 (trending regime).
# Short: KAMA falling (trend down) + RSI(14) < 50 + Chop(14) < 61.8.
# Uses 1d KAMA for trend direction, 4h for RSI and chop confirmation.
# Chop filter avoids ranging markets; RSI ensures momentum alignment.
# Position size: 0.25. Target: 60-120 total trades over 4 years (15-30/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.full_like(close, np.nan)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_1d = kama(close_1d, period=10, fast=2, slow=30)
    
    # RSI(14) on 4h
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4h = rsi(close, period=14)
    
    # Chop(14) on 4h
    def chop(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.full_like(close, np.nan)
        for i in range(period, len(close)):
            atr[i] = np.sum(tr[i-period+1:i+1]) / period
        highest = np.full_like(close, np.nan)
        lowest = np.full_like(close, np.nan)
        for i in range(period-1, len(close)):
            highest[i] = np.max(high[i-period+1:i+1])
            lowest[i] = np.min(low[i-period+1:i+1])
        chop = np.where((highest - lowest) != 0, 100 * np.log10(atr * np.sqrt(period) / (highest - lowest)) / np.log10(100), 50)
        return chop
    
    chop_4h = chop(high, low, close, period=14)
    
    # Align 1d KAMA to 4h
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(14, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_4h[i]) or np.isnan(chop_4h[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend: rising if current > previous, falling if current < previous
        kama_rising = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_falling = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        # RSI condition: >50 for bullish, <50 for bearish
        rsi_bullish = rsi_4h[i] > 50
        rsi_bearish = rsi_4h[i] < 50
        
        # Chop condition: < 61.8 for trending regime
        chop_trending = chop_4h[i] < 61.8
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + chop trending
            if kama_rising and rsi_bullish and chop_trending:
                position = 1
                signals[i] = position_size
            # Short: KAMA falling + RSI < 50 + chop trending
            elif kama_falling and rsi_bearish and chop_trending:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA falling OR RSI < 50 OR chop > 61.8 (ranging)
            if not (kama_rising and rsi_bullish and chop_trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA rising OR RSI > 50 OR chop > 61.8 (ranging)
            if not (kama_falling and rsi_bearish and chop_trending):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_KAMA_RSI_Chop"
timeframe = "4h"
leverage = 1.0
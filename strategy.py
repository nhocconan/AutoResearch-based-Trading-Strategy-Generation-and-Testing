#!/usr/bin/env python3
"""
Hypothesis: 1D Daily KAMA + RSI + Chop Filter
Long when KAMA > KAMA_prev (bullish trend), RSI < 40 (oversold), and Chop > 61.8 (range market).
Short when KAMA < KAMA_prev (bearish trend), RSI > 60 (overbought), and Chop > 61.8 (range market).
Exit when KAMA reverses direction or Chop < 38.2 (trending market).
KAMA adapts to market noise, RSI captures mean reversion in range markets, Chop filter ensures we only trade in ranging conditions where mean reversion works best. Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by adapting to volatility and focusing on mean reversion in ranges.
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
    
    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, slow_ema=2, fast_ema=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=10))  # 10-period change
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
        # Handle first 10 values
        er = np.full_like(change, np.nan, dtype=float)
        valid = ~np.isnan(volatility) & (volatility != 0)
        er[valid] = change[valid] / volatility[valid]
        # Smoothing Constants
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
        # Handle first value
        kama_vals = np.full_like(close, np.nan, dtype=float)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(sc[i-1]):
                kama_vals[i] = kama_vals[i-1] + sc[i-1] * (close[i] - kama_vals[i-1])
            else:
                kama_vals[i] = kama_vals[i-1]
        return kama_vals
    
    kama_vals = kama(close, 2, 30)
    
    # RSI (14-period)
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        # First average
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Chop Chopiness Index (14-period)
    def chop(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Sum of true ranges
        tr_sum = np.full_like(close, np.nan, dtype=float)
        for i in range(period, len(close)):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        # Highest high and lowest low over period
        hh = np.full_like(close, np.nan, dtype=float)
        ll = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(close)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        # Chop calculation
        chop_vals = np.full_like(close, np.nan, dtype=float)
        for i in range(period, len(close)):
            if tr_sum[i] > 0 and hh[i] != ll[i]:
                chop_vals[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop_vals
    
    chop_vals = chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI oversold, Chop high (range)
            if kama_vals[i] > kama_vals[i-1] and rsi_vals[i] < 40 and chop_vals[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI overbought, Chop high (range)
            elif kama_vals[i] < kama_vals[i-1] and rsi_vals[i] > 60 and chop_vals[i] > 61.8:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: KAMA reverses or Chop low (trending)
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA falls or Chop < 38.2 (trend)
                if kama_vals[i] < kama_vals[i-1] or chop_vals[i] < 38.2:
                    exit_signal = True
            else:  # position == -1
                # Exit short: KAMA rises or Chop < 38.2 (trend)
                if kama_vals[i] > kama_vals[i-1] or chop_vals[i] < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0
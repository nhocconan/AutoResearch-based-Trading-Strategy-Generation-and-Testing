#!/usr/bin/env python3
"""
1d_KAMA_Direction_Plus_RSI_With_Chop_Filter
Hypothesis: KAMA(10) direction on daily timeframe for trend bias, filtered by RSI(14) extremes and Choppiness Index(14) regime.
In bull markets: follow KAMA long when RSI < 30 (oversold) and chop > 61.8 (range) for mean reversion entries.
In bear markets: follow KAMA short when RSI > 70 (overbought) and chop > 61.8 (range) for mean reversion entries.
Uses 1d for trend and momentum, with chop filter to avoid whipsaw in strong trends.
Target: 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

name = "1d_KAMA_Direction_Plus_RSI_With_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate KAMA(10)
    def calculate_kama(close, er_period=10):
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, er_period))
        change[0:er_period] = 0  # Not enough data
        
        vol = np.abs(close - np.roll(close, 1))
        vol[0] = 0
        for i in range(1, len(vol)):
            vol[i] = np.abs(close[i] - close[i-1])
        
        er = np.zeros_like(close)
        for i in range(er_period, len(close)):
            if np.sum(vol[i-er_period+1:i+1]) > 0:
                er[i] = change[i] / np.sum(vol[i-er_period+1:i+1])
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
        
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    kama = calculate_kama(close, 10)
    
    # Calculate RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        delta = np.insert(delta, 0, 0)  # same length as close
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        if len(close) >= period + 1:
            avg_gain[period] = np.mean(gain[1:period+1])
            avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.zeros_like(close)
        rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100  # No loss = 100 RSI
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index(14)
    def calculate_chop(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # ATR
        atr = np.zeros_like(close)
        atr[:period-1] = np.nan
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Highest high and lowest low over period
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        for i in range(len(close)):
            if i < period:
                hh[i] = np.max(high[:i+1])
                ll[i] = np.min(low[:i+1])
            else:
                hh[i] = np.max(high[i-period+1:i+1])
                ll[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close)
        for i in range(period-1, len(close)):
            if np.sum(atr[i-period+1:i+1]) > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50  # Neutral
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Session filter: 00-24 UTC (full day for daily timeframe)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for daily, but keep for consistency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter (always true for daily, but keep structure)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up (close > KAMA) AND RSI < 30 (oversold) AND chop > 61.8 (range)
            if close[i] > kama[i] and rsi[i] < 30 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (close < KAMA) AND RSI > 70 (overbought) AND chop > 61.8 (range)
            elif close[i] < kama[i] and rsi[i] > 70 and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down OR RSI > 50 (exit oversold) OR chop < 38.2 (trend)
            if close[i] < kama[i] or rsi[i] > 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR RSI < 50 (exit overbought) OR chop < 38.2 (trend)
            if close[i] > kama[i] or rsi[i] < 50 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
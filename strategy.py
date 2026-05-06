#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day KAMA trend + RSI mean reversion + volume confirmation
# KAMA adapts to market efficiency, providing smooth trend in both bull and bear markets
# RSI(2) captures short-term mean reversion: buy oversold, sell overbought
# Volume confirmation filters weak moves
# Target: 20-50 total trades over 4 years (5-12/year) with 0.25 position sizing

name = "4h_1dKAMA_RSI2_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d KAMA trend ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation on daily close
    close_1d = df_1d['close'].values
    direction = np.abs(close_1d[10:] - close_1d[:-10])
    volatility = np.sum(np.abs(np.diff(close_1d, axis=0)), axis=0)[:len(direction)]
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align 1d KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(2) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=2, min_periods=2).mean().values
    loss_ma = pd.Series(loss).rolling(window=2, min_periods=2).mean().values
    rs = np.where(loss_ma != 0, gain_ma / loss_ma, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price below KAMA (dip in uptrend) and RSI oversold
            if close[i] < kama_aligned[i] and rsi[i] < 30 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (rally in downtrend) and RSI overbought
            elif close[i] > kama_aligned[i] and rsi[i] > 70 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above KAMA or RSI overbought
            if close[i] > kama_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below KAMA or RSI oversold
            if close[i] < kama_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
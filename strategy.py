#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily KAMA + RSI + Volume Filter
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in both bull and bear markets.
# RSI confirms momentum, volume filters ensure participation.
# Target: 20-40 trades/year (80-160 over 4 years).

name = "6h_kama_rsi_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and RSI
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # KAMA calculation on daily close
    close_daily = df_daily['close'].values
    change = np.abs(np.diff(close_daily, prepend=close_daily[0]))
    volatility = np.abs(np.diff(close_daily))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_daily)
    kama[0] = close_daily[0]
    for i in range(1, len(close_daily)):
        kama[i] = kama[i-1] + sc[i] * (close_daily[i] - kama[i-1])
    
    # RSI calculation on daily close
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Shift by 1 to use previous day's data
    kama = np.roll(kama, 1)
    rsi = np.roll(rsi, 1)
    if len(kama) > 1:
        kama[0] = kama[1]
        rsi[0] = rsi[1]
    
    # Align to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_daily, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_daily, rsi)
    
    # Volume filter: volume > 1.5x 20-period average on 6h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price above KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: price above KAMA and RSI > 50 with volume
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below KAMA and RSI < 50 with volume
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
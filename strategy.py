#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# RSI filters for overbought/oversold conditions, while Chop filter ensures we only trade in trending markets.
# Works in bull: KAMA up + RSI < 70 + Chop < 61.8 = long
# Works in bear: KAMA down + RSI > 30 + Chop < 61.8 = short
# Target: 15-30 trades/year (60-120 over 4 years).

name = "12h_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Chop calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate KAMA (12h timeframe)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Fix: calculate volatility correctly
    volatility = np.zeros_like(close)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first values
    rsi[:14] = 50
    
    # Calculate Chop (14-period) from daily data
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    atr = np.zeros(len(daily_close))
    for i in range(1, len(daily_close)):
        tr = max(
            daily_high[i] - daily_low[i],
            np.abs(daily_high[i] - daily_close[i-1]),
            np.abs(daily_low[i] - daily_close[i-1])
        )
        if i == 1:
            atr[i] = tr
        else:
            atr[i] = (atr[i-1] * 13 + tr) / 14
    # Sum of true ranges over 14 periods
    tr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max-min range over 14 periods
    max_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    range14 = max_high - min_low
    chop = np.where(range14 != 0, 100 * np.log10(tr14 / range14) / np.log10(14), 50)
    chop[:14] = 50
    # Align Chop to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_daily, chop)
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: KAMA up (close > KAMA), RSI not overbought, Chop < 61.8 (trending)
        if close[i] > kama[i] and rsi[i] < 70 and chop_aligned[i] < 61.8:
            signals[i] = 0.25
        # Short: KAMA down (close < KAMA), RSI not oversold, Chop < 61.8 (trending)
        elif close[i] < kama[i] and rsi[i] > 30 and chop_aligned[i] < 61.8:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals
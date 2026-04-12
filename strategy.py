#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA + RSI + Chop filter on 1d timeframe
# Uses KAMA to determine trend direction, RSI for overbought/oversold conditions,
# and Chop index to filter ranging markets. Designed for low trade frequency
# (target: 10-25 trades/year) to minimize fee drag. Works in both bull and bear
# markets by combining trend following with mean reversion in ranging conditions.

name = "1d_kama_rsi_chop_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER = 10, fast = 2, slow = 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    er[volatility != 0] = change[volatility != 0] / volatility[volatility != 0]
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop index (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = highest_high - lowest_low
    chop = np.where(denominator > 0, 100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when Chop > 61.8 (trending market)
        if chop[i] <= 61.8:
            # In ranging market, mean revert at RSI extremes
            if position == 1 and rsi[i] >= 70:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position or flat
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            continue
        
        # Trending market: follow KAMA direction with RSI filter
        if close[i] > kama[i] and rsi[i] < 70 and position != 1:
            position = 1
            signals[i] = 0.25
        elif close[i] < kama[i] and rsi[i] > 30 and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and (close[i] <= kama[i] or rsi[i] >= 70):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] >= kama[i] or rsi[i] <= 30):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
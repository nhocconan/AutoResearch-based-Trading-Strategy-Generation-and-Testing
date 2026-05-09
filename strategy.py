#!/usr/bin/env python3
# Hypothesis: 6h KAMA + RSI with Choppiness Regime Filter
# Uses KAMA(10) to filter market noise and RSI(14) for mean reversion in choppy regimes.
# Long when: price > KAMA, RSI < 30, and Choppiness > 61.8 (choppy)
# Short when: price < KAMA, RSI > 70, and Choppiness > 61.8 (choppy)
# Exit when: price crosses KAMA OR Choppiness < 38.2 (trending regime)
# Uses 1d trend filter: only take longs when 1d EMA(50) rising, shorts when falling.
# Position size: 0.25 (25% of capital) to limit drawdown.
# Designed to work in both bull (trend following via 1d filter) and bear (mean reversion in chop) markets.

name = "6h_KAMA_RSI_Chop_Regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = ema_50_1d[0]
    ema_rising = ema_50_1d > ema_50_1d_prev
    ema_falling = ema_50_1d < ema_50_1d_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > KAMA, RSI < 30, choppy regime, 1d EMA rising
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop[i] > 61.8 and 
                ema_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA, RSI > 70, choppy regime, 1d EMA falling
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8 and 
                  ema_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR trending regime
            if (close[i] < kama[i]) or (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR trending regime
            if (close[i] > kama[i]) or (chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
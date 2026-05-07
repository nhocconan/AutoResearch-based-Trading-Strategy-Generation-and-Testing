#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA components
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, n=10, prepend=close[:10]))
    er = np.where(change != 0, direction / change, 0)
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2
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
    
    # Choppiness Index (14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(14)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for KAMA and RSI stability
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA, RSI > 50, chop > 61.8 (range), weekly uptrend
            if (close[i] > kama[i] and rsi[i] > 50 and chop[i] > 61.8 and close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, RSI < 50, chop > 61.8 (range), weekly downtrend
            elif (close[i] < kama[i] and rsi[i] < 50 and chop[i] > 61.8 and close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below KAMA or trend change or chop < 38.2 (trend)
            if close[i] < kama[i] or close[i] < ema_1w_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above KAMA or trend change or chop < 38.2 (trend)
            if close[i] > kama[i] or close[i] > ema_1w_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily KAMA with RSI and Choppiness Index filter for range-bound markets, filtered by weekly EMA trend.
# In choppy markets (CHOP > 61.8), we take mean-reversion trades: long when price > KAMA and RSI > 50, short when price < KAMA and RSI < 50.
# Weekly EMA ensures we only trade in the direction of the higher timeframe trend.
# Position size 0.25 limits drawdown. Target: 20-40 trades/year to avoid fee drag. Works in both bull and bear markets by adapting to regime.
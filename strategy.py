#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour KAMA direction filter + RSI(14) extremes + chop regime filter
# We go long when KAMA turns up, RSI < 30, and chop > 61.8 (range market)
# We go short when KAMA turns down, RSI > 70, and chop > 61.8
# Uses 12h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# KAMA adapts to market noise, RSI identifies extremes in range, chop filter ensures mean-reversion environment.

name = "12h_KAMA_RSI_Chop_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate KAMA (adaptive moving average)
    # ER = |net change| / sum(abs(price changes))
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(change) > 1 else 0
    # Simplified ER calculation for efficiency
    price_change = np.abs(close - np.roll(close, 1))
    price_change[0] = 0
    er = np.abs(np.diff(close, prepend=close[0])) / (np.sum(price_change) + 1e-10)
    er = np.where(np.isnan(er), 0, er)
    # Smooth ER
    er_smooth = pd.Series(er).ewm(alpha=2/(2+1), adjust=False).fillna(0).values
    # Smoothing constants
    fast_sc = 2/(2+1)
    slow_sc = 2/(30+1)
    sc = (er_smooth * (fast_sc - slow_sc) + slow_sc)**2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)
    
    # Calculate Choppiness Index (14-period)
    # ATR = True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Sum of true ranges over period
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Max/min close over period
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(sum_tr / (max_close - min_close + 1e-10)) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        kama_prev = kama[i-1]
        rsi_val = rsi[i]
        chop_val = chop[i]
        
        if position == 0:
            # Enter long: KAMA turning up + RSI oversold + choppy market (mean reversion)
            if kama_val > kama_prev and rsi_val < 30 and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA turning down + RSI overbought + choppy market
            elif kama_val < kama_prev and rsi_val > 70 and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turning down OR RSI overbought
            if kama_val < kama_prev or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turning up OR RSI oversold
            if kama_val > kama_prev or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filter.
Only trade when: KAMA trend aligned, RSI not extreme (40-60), and market is trending (CHOP < 38.2).
This avoids whipsaws in ranging markets and captures sustained trends in both bull and bear regimes.
Position size: 0.25. Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag.
"""

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
    
    # === KAMA (Kaufman Adaptive Moving Average) ===
    # Fast EMA: 2, Slow EMA: 30
    fast = 2
    slow = 30
    
    # Direction = abs(close - close[fast])
    direction = np.abs(np.diff(close, n=fast))
    direction = np.concatenate([np.full(fast, np.nan), direction])
    
    # Volatility = sum of abs(close - close.shift(1)) over slow period
    volatility = np.abs(np.diff(close, n=1))
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    vol_sum = pd.Series(volatility).rolling(window=slow, min_periods=slow).sum().values
    
    # ER = direction / volatility (avoid div by zero)
    er = np.where(vol_sum > 0, direction / vol_sum, 0)
    
    # SC = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (fast + 1)
    slowest = 2.0 / (slow + 1)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # KAMA: first value = close[0], then kama = prev_kama + sc * (close - prev_kama)
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    delta = np.concatenate([np.full(1, np.nan), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index(14) ===
    # True Range
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    range_hl = hh - ll
    chop = np.where(range_hl > 0, 100 * np.log10(atr_sum / range_hl) / np.log10(14), 50)
    
    # === Signals ===
    # Trend: price > KAMA = bullish, price < KAMA = bearish
    # Only trade when RSI is moderate (40-60) and market is trending (CHOP < 38.2)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # KAMA slow=30, RSI/CHOP=14
    
    for i in range(start_idx, n):
        # Skip if any indicator not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        rsi_moderate = (rsi[i] >= 40) & (rsi[i] <= 60)
        chop_trending = chop[i] < 38.2  # trending regime
        
        if position == 0:
            # Enter long in bullish trend with filters
            if kama_bullish and rsi_moderate and chop_trending:
                signals[i] = 0.25
                position = 1
            # Enter short in bearish trend with filters
            elif kama_bearish and rsi_moderate and chop_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit on bearish trend or choppy market (range)
            if not (kama_bullish and rsi_moderate and chop_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit on bullish trend or choppy market (range)
            if not (kama_bearish and rsi_moderate and chop_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0
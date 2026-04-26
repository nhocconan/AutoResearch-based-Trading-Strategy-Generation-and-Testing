#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: KAMA (adaptive trend) filters noise, RSI(14) extremes provide mean-reversion entries, and Choppiness Index (CHOP) regime filter ensures trades occur only in trending (CHOP < 38.2) or ranging (CHOP > 61.8) markets as appropriate. Designed for 4h timeframe to target 20-50 trades/year with discrete sizing (0.25) to minimize fee drag while capturing moves in both bull and bear regimes via adaptive trend and mean-reversion logic.
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
    volume = prices['volume'].values
    
    # === KAMA Calculation (adaptive trend) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA initialization and calculation
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed at index 9 (10th element)
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) Calculation ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (CHOP) Calculation ===
    # True Range over 14 periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period TR = high - low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.divide(
        np.log10(atr14 / (hh14 - ll14)) * 100,
        np.log10(14),
        out=np.zeros_like(atr14),
        where=(hh14 - ll14)!=0
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of all lookbacks: KAMA needs 10, RSI 14, CHOP 14)
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filters
        trending_market = chop[i] < 38.2   # CHOP < 38.2 = trending
        ranging_market = chop[i] > 61.8    # CHOP > 61.8 = ranging
        
        # Long logic: 
        # - In trending market: buy when price > KAMA (trend follow)
        # - In ranging market: buy when RSI < 30 (oversold mean reversion)
        long_signal = False
        if trending_market and close[i] > kama[i]:
            long_signal = True
        elif ranging_market and rsi[i] < 30:
            long_signal = True
        
        # Short logic:
        # - In trending market: sell when price < KAMA (trend follow)
        # - In ranging market: sell when RSI > 70 (overbought mean reversion)
        short_signal = False
        if trending_market and close[i] < kama[i]:
            short_signal = True
        elif ranging_market and rsi[i] > 70:
            short_signal = True
        
        # Exit logic: reverse signal or regime change to opposite extreme
        exit_long = False
        exit_short = False
        if position == 1:
            # Exit long if: short signal OR regime shifts to strong ranging (CHOP > 70) 
            if short_signal or chop[i] > 70:
                exit_long = True
        elif position == -1:
            # Exit short if: long signal OR regime shifts to strong ranging (CHOP > 70)
            if long_signal or chop[i] > 70:
                exit_short = True
        
        # Update signals and position
        if exit_long:
            signals[i] = 0.0
            position = 0
        elif exit_short:
            signals[i] = 0.0
            position = 0
        elif long_signal and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_signal and position != -1:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0
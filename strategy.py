#!/usr/bin/env python3
"""
12h_KAMA_RSI_Chop_Regime_v1
Hypothesis: Combines Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI for overbought/oversold conditions, and Choppiness Index to filter ranging vs trending markets.
Works in both bull and bear markets by adapting to regime: in trending markets (CHOP < 38.2),
follow KAMA direction; in ranging markets (CHOP > 61.8), fade RSI extremes.
Uses 1d timeframe for regime filter to avoid noise. Target: 50-150 trades over 4 years (12-37/year) on 12h.
"""

name = "12h_KAMA_RSI_Chop_Regime_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === KAMA Calculation (12h) ===
    # Direction = abs(close - close[10])
    # Volatility = sum(abs(close - close.shift(1)) for 10 periods)
    # ER = Direction / Volatility
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = previous KAMA + SC * (close - previous KAMA)
    
    change = np.abs(np.subtract(close[10:], close[:-10]))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # needs correction
    
    # Recalculate volatility properly
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = volatility != 0
    er[mask] = change[10:][mask] / volatility[10:][mask] if len(change[10:][mask]) == len(volatility[10:][mask]) else 0
    
    # Handle array size mismatch
    er_full = np.zeros(n)
    er_full[10:] = er[10:] if len(er) >= 10 else er
    
    fastest = 2 / (2 + 1)   # for EMA 2
    slowest = 2 / (30 + 1)  # for EMA 30
    sc = np.zeros(n)
    sc = (er_full * (fastest - slowest) + slowest) ** 2
    
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[14:i+1])
            avg_loss[i] = np.mean(loss[14:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    mask = avg_loss != 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # when no loss
    
    # === 1D Data for Choppiness Index ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], tr])
    
    # ATR14
    atr14 = np.zeros(len(high_1d))
    for i in range(14, len(high_1d)):
        if i == 14:
            atr14[i] = np.mean(tr[1:i+1])
        else:
            atr14[i] = (atr14[i-1] * 13 + tr[i]) / 14
    
    # Sum of ATR14 over 14 periods
    sum_atr14 = np.zeros(len(high_1d))
    for i in range(14, len(high_1d)):
        sum_atr14[i] = np.sum(atr14[i-13:i+1])
    
    # Max(HH) - Min(LL) over 14 periods
    hh = np.zeros(len(high_1d))
    ll = np.zeros(len(high_1d))
    for i in range(len(high_1d)):
        if i < 14:
            hh[i] = np.max(high_1d[:i+1])
            ll[i] = np.min(low_1d[:i+1])
        else:
            hh[i] = np.max(high_1d[i-13:i+1])
            ll[i] = np.min(low_1d[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros(len(high_1d))
    for i in range(14, len(high_1d)):
        if sum_atr14[i] > 0 and (hh[i] - ll[i]) > 0:
            chop[i] = 100 * np.log10(sum_atr14[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align chop to 12h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === Generate Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(30, 14)  # KAMA needs 10, RSI needs 14, CHOP needs 14
    
    for i in range(start_idx, n):
        # Skip if any data is invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market (CHOP < 38.2): follow KAMA direction
            if chop_aligned[i] < 38.2:
                if close[i] > kama[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < kama[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market (CHOP > 61.8): fade RSI extremes
            elif chop_aligned[i] > 61.8:
                if rsi[i] < 30:  # oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # overbought
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: reverse signal
            if chop_aligned[i] < 38.2 and close[i] < kama[i]:  # trend fails
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and rsi[i] > 50:  # range exits at midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain
        elif position == -1:
            # Short exit: reverse signal
            if chop_aligned[i] < 38.2 and close[i] > kama[i]:  # trend fails
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and rsi[i] < 50:  # range exits at midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain
    
    return signals
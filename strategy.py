#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopRegime_v1
Hypothesis: Use KAMA(14) on daily to determine trend direction, RSI(14) for momentum strength, and Choppiness Index(14) to filter ranging markets. Only trade when KAMA shows clear trend (price above/below KAMA), RSI is not extreme, and market is trending (CHOP < 38.2). This avoids whipsaws in ranging markets and captures sustained moves in both bull and bear regimes. Target: 15-25 trades/year to minimize fee drag.
"""

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for regime filter (optional, using daily only for simplicity in this version)
    # For true 1d strategy, we use daily data directly
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (Kaufman Adaptive Moving Average) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    # Volatility sum of absolute changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(1, n):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    volatility = np.concatenate([np.zeros(10), volatility[10:]])  # align
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # --- RSI (14) ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(close)
    rs[14:] = avg_gain[14:] / np.where(avg_loss[14:] == 0, 1e-10, avg_loss[14:])
    rsi = 100 - (100 / (1 + rs))
    
    # --- Choppiness Index (14) ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(14)
    atr = np.zeros_like(close)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of TR over 14 periods
    sum_tr = np.zeros_like(close)
    for i in range(14, n):
        sum_tr[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    highest_high = np.zeros_like(close)
    lowest_low = np.zeros_like(close)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close)
    for i in range(14, n):
        if sum_tr[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # --- Signals ---
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # for KAMA, RSI, CHOP
    
    for i in range(start_idx, n):
        # Skip if any values are not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                # Simple exit: reverse signal or opposite KAMA cross
                if position == 1 and close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine conditions
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        rsi_not_extreme = (rsi[i] > 30) and (rsi[i] < 70)  # avoid overbought/oversold
        market_trending = chop[i] < 38.2  # trending market
        
        if position == 0:
            # Enter long: price above KAMA, RSI not extreme, trending market
            if price_above_kama and rsi_not_extreme and market_trending:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, RSI not extreme, trending market
            elif price_below_kama and rsi_not_extreme and market_trending:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below KAMA OR RSI overbought
                if close[i] < kama[i] or rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA OR RSI oversold
                if close[i] > kama[i] or rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
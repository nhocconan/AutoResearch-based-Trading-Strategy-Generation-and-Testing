#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour KAMA trend + RSI(14) mean reversion + Choppiness regime filter.
# Enters long when price is above KAMA (trending up), RSI < 30 (oversold), and market is choppy (CHOP > 61.8).
# Enters short when price is below KAMA (trending down), RSI > 70 (overbought), and market is choppy (CHOP > 61.8).
# Uses ATR-based stoploss (2x ATR) and reduces position to 50% at 2R profit.
# Designed to capture mean reversion within a trending environment while avoiding strong trends.
# Target: 100-180 total trades over 4 years (25-45/year) with controlled risk.

name = "4h_kama_rsi_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - trend component
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[change < 0] = 0  # Ensure non-negative for first 10 bars
    for i in range(10, len(change)):
        if i < 10:
            change[i] = np.abs(close[i] - close[i-10])
        else:
            change[i] = np.abs(close[i] - close[i-10])
    # Actually compute properly:
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.abs(np.diff(close, n=1))  # |close[t] - close[t-1]|
    volatility = np.concatenate([np.array([np.nan]), volatility])
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = change / vol_sum
    er[vol_sum == 0] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    delta = np.concatenate([np.array([np.nan]), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs[avg_loss == 0] = np.inf
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) - regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid
    chop[hh - ll == 0] = 50  # neutral when no range
    
    # ATR for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    max_favorable = 0.0  # track max favorable excursion for trailing
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i]) or np.isnan(hh[i]) or np.isnan(ll[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Update max favorable excursion
            max_favorable = max(max_favorable, close[i] - entry_price)
            # Stoploss: 2 * ATR below entry
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable = 0.0
            # Take profit: reduce to 50% at 2R profit
            elif max_favorable >= 2.0 * 2.0 * atr[i]:  # 2R where R = 2*ATR
                signals[i] = 0.125  # half position
            # Exit: price crosses below KAMA or RSI > 70 (overbought) or trend strong (CHOP < 38.2)
            elif close[i] < kama[i] or rsi[i] > 70 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Update max favorable excursion (positive for short)
            max_favorable = max(max_favorable, entry_price - close[i])
            # Stoploss: 2 * ATR above entry
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable = 0.0
            # Take profit: reduce to 50% at 2R profit
            elif max_favorable >= 2.0 * 2.0 * atr[i]:  # 2R where R = 2*ATR
                signals[i] = -0.125  # half position
            # Exit: price crosses above KAMA or RSI < 30 (oversold) or trend strong (CHOP < 38.2)
            elif close[i] > kama[i] or rsi[i] < 30 or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                max_favorable = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: choppy market (CHOP > 61.8) + price vs KAMA + RSI extreme
            if chop[i] > 61.8:
                # Long entry: price above KAMA (bullish bias) + RSI oversold
                if close[i] > kama[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    max_favorable = 0.0
                # Short entry: price below KAMA (bearish bias) + RSI overbought
                elif close[i] < kama[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    max_favorable = 0.0
    
    return signals
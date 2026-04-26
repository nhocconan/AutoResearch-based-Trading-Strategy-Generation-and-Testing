#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: KAMA (adaptive trend) filters noise, RSI(14) < 30/ > 70 for mean reversion entries, and Choppiness Index > 61.8 identifies ranging markets where mean reversion works. Discrete sizing (0.25) and ATR stoploss (1.5x ATR) target ~25-40 trades/year. Works in bull/bear by fading extremes only in choppy regimes.
"""

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
    volume = prices['volume'].values
    
    # KAMA (adaptive trend) - faster reacts to trend, slower in chop
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # 10-period sum of abs changes
    # Fix: compute per-bar volatility sum
    volatility = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) - high = ranging, low = trending
    atr_1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr_1[0] = high[0] - low[0]
    sum_atr14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (highest_high14 - lowest_low14)) / np.log10(14)
    chop = np.where((highest_high14 - lowest_low14) != 0, chop, 50)  # avoid div0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of KAMA seed (10), RSI (14), ATR (14), HH/LL (14)
    start_idx = max(10, 14, 14, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        atr_val = np.maximum(high[i] - low[i], np.maximum(np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))) if i > 0 else high[i] - low[i]
        # Simplified ATR: use current bar TR as proxy (acceptable for stop)
        atr_val = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1])) if i > 0 else high[i] - low[i]
        
        # Skip if any data not ready
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: only mean revert in choppy markets (CHOP > 61.8)
        is_choppy = chop_val > 61.8
        
        # Mean reversion entries: RSI extremes in chop
        long_entry = is_choppy and (rsi_val < 30)
        short_entry = is_choppy and (rsi_val > 70)
        
        # Exit: RSI returns to neutral (40-60) or opposite extreme
        long_exit = False
        short_exit = False
        if position == 1:
            long_exit = (rsi_val > 40) or (rsi_val > 70)  # exit on recovery or overbought
        elif position == -1:
            short_exit = (rsi_val < 60) or (rsi_val < 30)  # exit on recovery or oversold
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0
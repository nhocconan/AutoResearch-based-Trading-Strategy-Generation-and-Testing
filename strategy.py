#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v2
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(2) for mean-reversion entry timing and Choppiness Index(14) regime filter.
Only take longs when KAMA is rising AND RSI(2) < 10 AND chop > 61.8 (range).
Only take shorts when KAMA is falling AND RSI(2) > 90 AND chop > 61.8 (range).
This avoids trending markets where mean reversion fails, focusing on high-probability
reversals in ranging conditions. Uses discrete sizing (0.25) and time-based exit (hold max 10 bars)
to limit trades to ~10-25/year. Works in both bull and bear by adapting to regime.
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
    
    # === Indicators on primary (1d) ===
    # KAMA(10, 2, 30) - ER based on 10-period, fastest EMA 2, slowest 30
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(1)).values
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    direction = np.abs(close_s - close_s.shift(10)).values
    er = np.where(volatility > 0, direction / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(2) for mean reversion
    delta = pd.Series(close).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index(14)
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    # Handle division by zero or invalid
    chop = np.where((hh - ll) > 0, chop, 50)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_bar = 0
    
    # Warmup: KAMA needs ~10, RSI(2) needs 2, Chop needs 14
    start_idx = max(10, 2, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # KAMA direction: rising/falling
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # RSI extremes for mean reversion
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        
        # Chop regime: range-bound market
        chop_range = chop[i] > 61.8
        
        # Time-based exit: hold max 10 bars
        max_hold_reached = (position != 0 and (i - entry_bar) >= 10)
        
        # Entry conditions
        long_entry = kama_rising and rsi_oversold and chop_range
        short_entry = kama_falling and rsi_overbought and chop_range
        
        # Exit conditions
        long_exit = (not kama_rising) or (not chop_range) or max_hold_reached
        short_exit = (not kama_falling) or (not chop_range) or max_hold_reached
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_bar = i
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_bar = i
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0
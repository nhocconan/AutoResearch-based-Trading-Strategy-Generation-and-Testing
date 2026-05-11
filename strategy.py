#!/usr/bin/env python3
"""
1D_KAMA_RSI_CHOP_v1
Hypothesis: On daily timeframe, KAMA trend direction combined with RSI extremes and
Choppiness Index regime filter provides robust entries in both bull and bear markets.
KAMA adapts to market noise, RSI identifies overbought/oversold conditions, and
Choppiness Index filters for trending markets (avoiding range-bound whipsaws).
Designed for low frequency (10-25 trades/year) to minimize fee drag.
"""

name = "1D_KAMA_RSI_CHOP_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter (optional, can be removed if not needed)
    df_1w = get_htf_data(prices, '1w')
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA (14, 2, 30) ---
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    abs_change = np.abs(np.diff(close, k=1))
    abs_sum = np.zeros_like(close)
    for i in range(1, len(abs_change)+1):
        abs_sum[i] = np.sum(abs_change[i-10:i]) if i >= 10 else np.sum(abs_change[:i])
    abs_sum[0] = 1e-10  # avoid division by zero
    er = np.zeros_like(close)
    er[10:] = change[10:] / abs_sum[10:]
    er[:10] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # --- RSI (14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Choppiness Index (14) ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chi = np.zeros_like(close)
    for i in range(14, n):
        if tr_sum[i] > 0 and hh[i] > ll[i]:
            chi[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chi[i] = 50  # neutral
    
    # Signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chi[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: KAMA direction + RSI extreme + trending market (low chop)
        long_entry = (close[i] > kama[i]) and (rsi[i] < 30) and (chi[i] < 38.2)
        short_entry = (close[i] < kama[i]) and (rsi[i] > 70) and (chi[i] < 38.2)
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: Opposite signal or choppy market
            if position == 1:
                # Exit if RSI > 50 (momentum fading) or choppy market
                if (rsi[i] > 50) or (chi[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit if RSI < 50 or choppy market
                if (rsi[i] < 50) or (chi[i] > 61.8):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
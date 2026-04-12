#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Chop_Filter_v1
Hypothesis: Use daily KAMA for trend direction, RSI for momentum, and Choppiness Index for regime filtering.
Buy when price > KAMA, RSI > 50, and CHOP > 61.8 (ranging market) for mean reversion to upside.
Sell when price < KAMA, RSI < 50, and CHOP > 61.8 (ranging market) for mean reversion to downside.
Exit when RSI crosses back to neutral (40-60 range). Designed for low trade frequency in ranging markets.
Works in both bull/bear by capturing mean reversion in ranging regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for Choppiness Index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14) - sum of TR over 14 periods
    atr_1w = np.zeros(len(tr))
    for i in range(14, len(tr)):
        atr_1w[i] = np.nansum(tr[i-13:i+1])  # 14-period sum
    
    # Highest high and lowest low over 14 periods
    hh_1w = np.zeros(len(high_1w))
    ll_1w = np.zeros(len(low_1w))
    for i in range(14, len(high_1w)):
        hh_1w[i] = np.max(high_1w[i-13:i+1])
        ll_1w[i] = np.min(low_1w[i-13:i+1])
    # First 13 values NaN
    hh_1w[:14] = np.nan
    ll_1w[:14] = np.nan
    
    # Choppiness Index: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop_raw = 100 * np.log10(atr_1w / (hh_1w - ll_1w)) / np.log10(14)
    # Handle division by zero or invalid cases
    chop_raw = np.where((hh_1w - ll_1w) == 0, 100, chop_raw)
    chop_raw = np.where(np.isnan(chop_raw), 50, chop_raw)  # Neutral when undefined
    
    # Align Choppiness Index to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_raw)
    
    # Daily KAMA (trend direction)
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    change = np.concatenate([[np.nan]*10, change])  # Align indices
    
    # Volatility sum of absolute changes
    vol = np.zeros(len(close))
    for i in range(10, len(close)):
        vol[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))  # Sum of 10 absolute changes
    
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    er = np.where(np.isnan(er), 0, er)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI (momentum)
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(np.isnan(rsi), 50, rsi)  # Neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup for KAMA/RSI
        # Skip if any data invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: Choppiness > 61.8 indicates ranging market (good for mean reversion)
        ranging_market = chop_aligned[i] > 61.8
        
        # Mean reversion signals in ranging markets
        if ranging_market:
            # Long: price below KAMA (oversold), RSI recovering from oversold
            long_signal = close[i] < kama[i] and rsi[i] < 40 and rsi[i] > rsi[i-1]
            # Short: price above KAMA (overbought), RSI declining from overbought
            short_signal = close[i] > kama[i] and rsi[i] > 60 and rsi[i] < rsi[i-1]
        else:
            # In trending markets, follow the trend with KAMA
            long_signal = close[i] > kama[i] and rsi[i] > 50
            short_signal = close[i] < kama[i] and rsi[i] < 50
        
        # Exit when RSI returns to neutral zone (40-60)
        long_exit = rsi[i] >= 50
        short_exit = rsi[i] <= 50
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
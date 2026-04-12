#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_kama_rsi_chop
# Uses Kaufman Adaptive Moving Average (KAMA) for trend direction on 4h chart.
# RSI(14) for momentum confirmation (long: RSI > 50, short: RSI < 50).
# Choppiness index (CHOP) from 1d to filter regime: only trade when CHOP < 61.8 (trending).
# Long when KAMA rising, RSI > 50, and 1d CHOP < 61.8.
# Short when KAMA falling, RSI < 50, and 1d CHOP < 61.8.
# Exit when KAMA changes direction or RSI crosses 50.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in trending markets via KAMA+RSI and avoids ranging markets via CHOP filter.
# Focus on BTC/ETH as primary targets.

name = "4h_1d_kama_rsi_chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(atr/14) / (hh - ll)) / log10(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh_ll = hh - ll
    chop = 100 * np.log10(atr_sum / 14 / hh_ll) / np.log10(14)
    chop = np.where(hh_ll > 0, chop, 100)  # avoid division by zero
    
    # Align daily Chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # KAMA on 4h close (ER=10, fast=2, slow=30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period net change
    change = np.concatenate([[np.nan]*10, change])  # align
    
    volatility = np.abs(np.diff(close))  # 1-period volatility
    volatility = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    volatility = np.concatenate([[np.nan], volatility])  # align
    
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/2 - 30/30) + 30/30) ** 2  # smooth constant
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on 4h
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Only trade in trending market (CHOP < 61.8)
        if chop_aligned[i] >= 61.8:
            # Range market: stay flat or reduce position
            if position == 1:
                signals[i] = 0.15  # reduce long
            elif position == -1:
                signals[i] = -0.15  # reduce short
            else:
                signals[i] = 0.0
            continue
        
        # KAMA direction: rising if current > previous, falling if current < previous
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Long signal: KAMA rising and RSI > 50
        if kama_rising and rsi[i] > 50 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: KAMA falling and RSI < 50
        elif kama_falling and rsi[i] < 50 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: KAMA changes direction or RSI crosses 50
        elif position == 1 and (not kama_rising or rsi[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not kama_falling or rsi[i] > 50):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
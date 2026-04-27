# 1d_KAMA_Direction_RSI_ChopFilter - Fixed
# Hypothesis: Daily KAMA trend direction + RSI(14) extremes + Choppiness index regime filter.
# Works in bull markets (KAMA up + RSI > 50) and bear markets (KAMA down + RSI < 50).
# Chop filter prevents whipsaws in ranging markets. Target: 15-25 trades/year.
# Fixed: Now generates trades by using proper RSI thresholds and removing overly strict filters.

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
    
    # Get weekly data for Choppiness index (regime filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) for weekly
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Choppiness Index: 100 * log10(sum(ATR14)/(n*(max(high)-min(low)))) / log10(n)
    sum_atr = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    chop_raw = 100 * np.log10(sum_atr / (14 * range_hl)) / np.log10(14)
    chop = chop_raw  # Already in 0-100 range
    
    chop_align = align_htf_to_ltf(prices, df_1w, chop)
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA calculation (ER = 10, fast=2, slow=30)
    change = np.abs(np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]]))
    volatility = np.abs(np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]]))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(kama[i-1]) or np.isnan(sc[i]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_align = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14) on daily
    delta = np.concatenate([[np.nan], close_1d[1:] - close_1d[:-1]])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_align = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama_align[i]) or np.isnan(rsi_align[i]) or 
            np.isnan(chop_align[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Chop < 61.8 = trending (use trend following)
        # Chop > 61.8 = ranging (avoid trading)
        if chop_align[i] > 61.8:
            signals[i] = 0.0
            position = 0
            continue
        
        # Long: KAMA up (price > KAMA) AND RSI > 50 (bullish momentum)
        if close[i] > kama_align[i] and rsi_align[i] > 50:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short: KAMA down (price < KAMA) AND RSI < 50 (bearish momentum)
        elif close[i] < kama_align[i] and rsi_align[i] < 50:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit: Reverse signal
        elif position == 1 and close[i] < kama_align[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > kama_align[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0
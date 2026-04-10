#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(14) mean reversion + choppiness regime filter
# - KAMA(10,2,30) from 1d: trend direction (KAMA rising = long bias, falling = short bias)
# - RSI(14) from 1d: long when RSI < 30 (oversold), short when RSI > 70 (overbought)
# - Choppiness Index(14) from 1w: only trade when CHOP > 61.8 (ranging market) for mean reversion
# - Designed for 1d timeframe: targets 15-30 trades/year to avoid fee drag
# - Works in ranging markets: chop filter ensures we mean revert in sideways action
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Volatility-adjusted exit: reverse signal when RSI crosses 50 (mean reversion complete)

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for chop filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d KAMA(10,2,30) for trend direction
    close_1d = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, 10))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=1)
    er = np.zeros_like(close_1d)
    er[10:] = change / (volatility + 1e-10)
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close_1d)
    kama[9] = close_1d[9]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    
    # Pre-compute 1d RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Pre-compute 1w Choppiness Index(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = 100 * np.log10(tr_sum_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    chop = np.concatenate([np.full(13, np.nan), chop])
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_rising[i]) or np.isnan(kama_falling[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in ranging markets (chop > 61.8)
        if chop_aligned[i] <= 61.8:
            # Exit position if chop becomes too low (trending market)
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries aligned with KAMA trend
            if kama_rising[i] and rsi[i] < 30:
                # Long bias + oversold = long
                position = 1
                signals[i] = 0.25
            elif kama_falling[i] and rsi[i] > 70:
                # Short bias + overbought = short
                position = -1
                signals[i] = -0.25
    
    return signals
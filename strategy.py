#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market regime, RSI identifies extremes, chop filter avoids whipsaws.
# Works in bull via KAMA trend following, in bear via mean reversion at RSI extremes when chop high.
# Target: 10-25 trades/year to minimize fee drag on 1d timeframe.
name = "daily_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA(10) on close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 1.0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chop(14) from weekly data
    # True Range
    tr1 = df_1w['high'].values[1:] - df_1w['low'].values[1:]
    tr2 = np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1])
    tr3 = np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
    tr_w = np.concatenate([[np.max([df_1w['high'].values[0] - df_1w['low'].values[0], 
                                   np.abs(df_1w['high'].values[0] - df_1w['close'].values[0]), 
                                   np.abs(df_1w['low'].values[0] - df_1w['close'].values[0])])], 
                          np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    ll_w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    chop_w = 100 * np.log10(hh_w - ll_w) / np.log10(14) / np.log10(np.sum(atr_w, axis=0))
    # Fix: sum over rolling window
    atr_sum_w = pd.Series(tr_w).rolling(window=14, min_periods=14).sum().values
    chop_w = 100 * np.log10(hh_w - ll_w) / np.log10(14) / np.log10(atr_sum_w)
    chop_w = np.where(atr_sum_w > 0, chop_w, 50)  # default to neutral chop
    
    # Align weekly chop to daily
    chop_w_aligned = align_htf_to_ltf(prices, df_1w, chop_w)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when chop > 61.8 (ranging market) for mean reversion
        # or when chop < 38.2 (trending) for trend following
        chop = chop_w_aligned[i]
        is_ranging = chop > 61.8
        is_trending = chop < 38.2
        
        if is_ranging:
            # Mean reversion in ranging markets
            if close[i] > kama[i] and rsi[i] < 30:
                signals[i] = 0.25  # Long
            elif close[i] < kama[i] and rsi[i] > 70:
                signals[i] = -0.25  # Short
            else:
                signals[i] = 0.0
        elif is_trending:
            # Trend following in trending markets
            if close[i] > kama[i]:
                signals[i] = 0.25  # Long
            elif close[i] < kama[i]:
                signals[i] = -0.25  # Short
            else:
                signals[i] = 0.0
        else:
            # Neutral chop - no trade
            signals[i] = 0.0
    
    return signals
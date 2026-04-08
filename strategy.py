#!/usr/bin/env python3
# 12h_1d_1w_kama_rsi_chop_regime_v1
# Hypothesis: 12h KAMA trend with RSI momentum filter and 1d/1w choppiness regime filter.
# Long: KAMA rising, RSI > 50, and 1d/1w CHOP > 61.8 (range regime) → mean reversion long from oversold
# Short: KAMA falling, RSI < 50, and 1d/1w CHOP > 61.8 (range regime) → mean reversion short from overbought
# Uses 12h primary timeframe with 1d/1w HTF for choppiness regime filter.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull/bear markets by adapting to range regimes where mean reversion performs best.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_kama_rsi_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h KAMA (adaptive trend)
    # Efficiency Ratio over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 12h RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (14)
    atr_1d = np.zeros(len(df_1d))
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    for i in range(1, len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        else:
            if np.isnan(atr_1d[i-1]):
                atr_1d[i] = np.nanmean(tr_1d[i-13:i+1])
            else:
                atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Highest high and lowest low over 14 periods
    hh_1d = np.full(len(df_1d), np.nan)
    ll_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 13:
            hh_1d[i] = np.max(high_1d[i-13:i+1])
            ll_1d[i] = np.min(low_1d[i-13:i+1])
    
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if not np.isnan(hh_1d[i]) and not np.isnan(ll_1d[i]) and hh_1d[i] > ll_1d[i]:
            sum_atr = np.nansum(atr_1d[i-13:i+1]) if i >= 13 else np.nan
            if not np.isnan(sum_atr) and sum_atr > 0:
                chop_1d[i] = 100 * np.log10(sum_atr / (hh_1d[i] - ll_1d[i])) / np.log10(14)
    
    # Get 1w data for choppiness index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Choppiness Index (14)
    atr_1w = np.zeros(len(df_1w))
    tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                       np.maximum(np.abs(high_1w[1:] - close_1w[:-1]),
                                  np.abs(low_1w[1:] - close_1w[:-1])))
    tr_1w = np.concatenate([[np.nan], tr_1w])
    for i in range(1, len(tr_1w)):
        if i < 14:
            atr_1w[i] = np.nan
        else:
            if np.isnan(atr_1w[i-1]):
                atr_1w[i] = np.nanmean(tr_1w[i-13:i+1])
            else:
                atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Highest high and lowest low over 14 periods
    hh_1w = np.full(len(df_1w), np.nan)
    ll_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if i >= 13:
            hh_1w[i] = np.max(high_1w[i-13:i+1])
            ll_1w[i] = np.min(low_1w[i-13:i+1])
    
    chop_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        if not np.isnan(hh_1w[i]) and not np.isnan(ll_1w[i]) and hh_1w[i] > ll_1w[i]:
            sum_atr = np.nansum(atr_1w[i-13:i+1]) if i >= 13 else np.nan
            if not np.isnan(sum_atr) and sum_atr > 0:
                chop_1w[i] = 100 * np.log10(sum_atr / (hh_1w[i] - ll_1w[i])) / np.log10(14)
    
    # Align 1d and 1w chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if not enough data
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(chop_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: both 1d and 1w must be in range (CHOP > 61.8)
        if chop_1d_aligned[i] <= 61.8 or chop_1w_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # Mean reversion signals in range market
        # Long: KAMA rising (trend up) but RSI not overbought -> pullback long
        # Short: KAMA falling (trend down) but RSI not oversold -> pullback short
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if kama_rising and rsi[i] < 50:  # Pullback in uptrend
            signals[i] = 0.25
        elif kama_falling and rsi[i] > 50:  # Pullback in downtrend
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals
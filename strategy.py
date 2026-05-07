#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    volatility = np.concatenate([np.full(er_len-1, np.nan), volatility[er_len-1:]])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (weekly)
    df_1w = get_htf_data(prices, '1w')
    chop = np.full(n, np.nan)
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        atr_1w = np.full(len(high_1w), np.nan)
        tr_1w = np.maximum(high_1w[1:] - low_1w[1:], 
                           np.maximum(np.abs(high_1w[1:] - np.concatenate([[np.nan], high_1w[:-1]])),
                                      np.abs(low_1w[1:] - np.concatenate([[np.nan], low_1w[:-1]]))))
        atr_1w[1:] = tr_1w
        for i in range(14, len(atr_1w)):
            if not np.isnan(atr_1w[i-14:i]).any():
                atr_1w[i] = np.nansum(atr_1w[i-14:i])
        highest_1w = np.full(len(high_1w), np.nan)
        lowest_1w = np.full(len(low_1w), np.nan)
        for i in range(14, len(high_1w)):
            highest_1w[i] = np.nanmax(high_1w[i-14:i])
            lowest_1w[i] = np.nanmin(low_1w[i-14:i])
        chop_1w = 100 * np.log10(atr_1w[14:] / (highest_1w[14:] - lowest_1w[14:])) / np.log10(14)
        chop_1w = np.concatenate([np.full(14, np.nan), chop_1w])
        chop = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Signals
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(er_len + 1, 14)
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        chop_high = chop[i] > 61.8  # ranging
        chop_low = chop[i] < 38.2   # trending
        
        if position == 0:
            if kama_up and rsi_oversold and chop_high:
                signals[i] = 0.25
                position = 1
            elif kama_down and rsi_overbought and chop_high:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            if not kama_up or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if not kama_down or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets. Combined with RSI extremes and weekly Chop filter, it avoids false signals in strong trends while capturing reversals in ranging markets. Works in bull markets (KAMA up + RSI oversold in chop) and bear markets (KAMA down + RSI overbought in chop). Discrete sizing (0.25) limits drawdown and reduces trade frequency. Target: 20-50 trades over 4 years (5-12.5/year) to minimize fee drag. Weekly Chop filter ensures entries only occur in ranging conditions where mean reversion is effective, avoiding whipsaws in strong trends. This addresses the failure of pure trend/follow strategies in 2022 chop and pure mean reversion in strong trends. Uses weekly timeframe for Chop to avoid noise and ensure regime stability. 1d primary timeframe balances responsiveness with cost control.
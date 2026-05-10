#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
# Combined with RSI for momentum confirmation and Choppiness Index to avoid false signals in low-volatility chop,
# this strategy aims to capture sustained moves while minimizing whipsaws. Works in bull markets by following
# upward KAMA slope and in bear markets by following downward slope, with volatility filter to avoid ranging conditions.

name = "4h_KAMA_Trend_With_RSI_and_Chop_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER = |Close - Close(past 10)| / Sum|Close - Close(past 1)| over 10 periods
    # SSC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prior KAMA + SSC * (Close - prior KAMA)
    close_1d = df_1d['close'].values
    if len(close_1d) < 10:
        return np.zeros(n)
    
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(change[:10]) if len(change) >= 10 else np.sum(change)
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        price_change = np.abs(close_1d[i] - close_1d[i-10])
        volatility = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        er[i] = price_change / volatility if volatility > 0 else 0
    
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    ss = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[9] = close_1d[9]  # Initialize
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + ss[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on daily close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = np.concatenate([np.full(13, np.nan), rsi])
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Choppiness Index on daily data
    # CI = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    atr_data = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr1 = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        tr2 = np.abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1])
        tr3 = np.abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        atr_data[i] = max(tr1, tr2, tr3)
    
    atr_sum = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        atr_sum[i] = np.sum(atr_data[i-13:i+1])
    
    max_high = np.zeros(len(df_1d))
    min_low = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        max_high[i] = np.max(df_1d['high'].iloc[i-13:i+1])
        min_low[i] = np.min(df_1d['low'].iloc[i-13:i+1])
    
    chop = np.zeros(len(df_1d))
    for i in range(14, len(df_1d)):
        if max_high[i] - min_low[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral when no range
    chop_1d = chop
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10), RSI (14), Chop (14), Vol MA (20)
    start_idx = max(10, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price vs KAMA
        above_kama = close[i] > kama_1d_aligned[i]
        below_kama = close[i] < kama_1d_aligned[i]
        
        # Momentum filter: RSI
        rsi_bull = rsi_1d_aligned[i] > 50
        rsi_bear = rsi_1d_aligned[i] < 50
        
        # Chop filter: avoid low volatility chop (Chop > 61.8 = ranging)
        chop_low = chop_1d_aligned[i] < 61.8
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: above KAMA + RSI bull + not chop + volume
            if above_kama and rsi_bull and chop_low and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: below KAMA + RSI bear + not chop + volume
            elif below_kama and rsi_bear and chop_low and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: below KAMA or RSI turns bear or chop high
            if not above_kama or not rsi_bull or not chop_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: above KAMA or RSI turns bull or chop high
            if not below_kama or not rsi_bear or not chop_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
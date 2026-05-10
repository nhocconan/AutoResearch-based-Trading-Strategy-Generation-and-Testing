#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# Combine with RSI for momentum confirmation and volume filter to reduce false signals.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (12-37/year) to minimize drag while capturing sustained moves.

name = "12h_KAMA_Trend_RSI_Filter"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA40 for trend filter
    ema_40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Calculate KAMA (adaptive moving average) on daily
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'])).rolling(window=10, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(df_1d['close'], np.nan, dtype=float)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_values = kama
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_values)
    
    # Calculate RSI(14) on daily
    delta = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Volume confirmation (24-period MA on 12h = ~12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA40 (40), KAMA/RSI (30), volume MA (24)
    start_idx = max(40, 30, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_40_1w_aligned[i]) or 
            np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_40_1w_aligned[i]
        weekly_downtrend = close[i] < ema_40_1w_aligned[i]
        
        # KAMA trend (price above/below KAMA)
        kama_uptrend = close[i] > kama_aligned[i]
        kama_downtrend = close[i] < kama_aligned[i]
        
        # RSI momentum (avoid extreme overbought/oversold)
        rsi_momentum_long = (rsi_aligned[i] > 50) and (rsi_aligned[i] < 70)
        rsi_momentum_short = (rsi_aligned[i] < 50) and (rsi_aligned[i] > 30)
        
        # Volume confirmation (>1.5x MA)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: weekly uptrend + price above KAMA + RSI bullish + volume
            if weekly_uptrend and kama_uptrend and rsi_momentum_long and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend + price below KAMA + RSI bearish + volume
            elif weekly_downtrend and kama_downtrend and rsi_momentum_short and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR price crosses below KAMA
            if not weekly_uptrend or not kama_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend turns up OR price crosses above KAMA
            if not weekly_downtrend or not kama_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
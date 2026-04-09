#!/usr/bin/env python3
# 1d_kama_rsi_chop_v1
# Hypothesis: Daily strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
# combined with RSI for momentum confirmation and Choppiness Index for regime filtering.
# Long when: KAMA trending up, RSI > 50, Chop < 61.8 (trending market).
# Short when: KAMA trending down, RSI < 50, Chop < 61.8 (trending market).
# Exit: Opposite KAMA signal or Chop > 61.8 (range market) or RSI divergence.
# Uses 1-week EMA200 for higher timeframe trend filter to avoid counter-trend trades.
# Designed for low trade frequency (target: 7-25 trades/year) to minimize fee drag.
# Works in both bull and bear markets via regime filter and higher timeframe trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Kaufman Adaptive Moving Average (KAMA) - 10 period
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0)  # temporary fix
    # Recompute volatility properly
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14 period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first value
    rsi = np.concatenate([[np.nan], rsi])
    
    # Choppiness Index (14 period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Prepend first TR
    tr = np.concatenate([[tr[0]], tr])
    # Sum of TR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where((hh - ll) != 0, 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14), 50)
    
    # 1-week EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: slope over 2 periods
        kama_up = kama[i] > kama[i-2]
        kama_down = kama[i] < kama[i-2]
        
        # Regime filter: trending market (Chop < 61.8)
        trending_market = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR Chop > 61.8 (range) OR RSI < 50 (momentum loss)
            if (not kama_up) or (not trending_market) or (rsi[i] < 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR Chop > 61.8 (range) OR RSI > 50 (momentum loss)
            if (not kama_down) or (not trending_market) or (rsi[i] > 50):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: KAMA up, RSI > 50, trending market, price above weekly EMA200
            if (kama_up and rsi[i] > 50 and trending_market and close[i] > ema200_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA down, RSI < 50, trending market, price below weekly EMA200
            elif (kama_down and rsi[i] < 50 and trending_market and close[i] < ema200_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
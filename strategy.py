#!/usr/bin/env python3

"""
Hypothesis: Daily KAMA direction with RSI(14) and choppiness filter.
Trades in direction of KAMA trend only when RSI is not extreme (30-70) and market is not choppy (CHOP > 61.8).
Avoids whipsaws in sideways markets and captures trending moves.
Targets 7-25 trades/year (30-100 total over 4 years) with disciplined entry.
Uses weekly trend filter to avoid counter-trend trades in strong trends.
"""

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
    
    # Load daily data for KAMA and RSI - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, k=1)), axis=0)  # 1-period volatility sum
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    
    # RSI(14) on daily
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first value
    rsi = np.concatenate([np.array([np.nan]), rsi])
    
    # Choppiness Index on daily (14-period)
    atr1 = np.abs(high - low)
    atr2 = np.abs(high - np.roll(close, 1))
    atr3 = np.abs(low - np.roll(close, 1))
    atr1[0] = 0
    atr2[0] = 0
    atr3[0] = 0
    tr = np.maximum(atr1, np.maximum(atr2, atr3))
    # Sum of true ranges over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness formula
    chop = np.where(tr_sum > 0, 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14), 50)
    # Handle division by zero when hh == ll
    chop = np.where((hh - ll) == 0, 50, chop)
    
    # Align daily indicators to 15m timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        
        # RSI filter: not overbought/oversold
        rsi_ok = (rsi_aligned[i] > 30) and (rsi_aligned[i] < 70)
        
        # Chop filter: not choppy (trending market)
        chop_ok = chop_aligned[i] > 61.8
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Enter long: price above KAMA, RSI OK, not choppy, weekly uptrend
            if price_above_kama and rsi_ok and chop_ok and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, RSI OK, not choppy, weekly downtrend
            elif price_below_kama and rsi_ok and chop_ok and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below KAMA or RSI overbought or choppy
                if (close[i] < kama_aligned[i] or 
                    rsi_aligned[i] >= 70 or 
                    chop_aligned[i] <= 61.8):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above KAMA or RSI oversold or choppy
                if (close[i] > kama_aligned[i] or 
                    rsi_aligned[i] <= 30 or 
                    chop_aligned[i] <= 61.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_1wEMA34"
timeframe = "1d"
leverage = 1.0
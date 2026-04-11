#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: 1d KAMA trend with RSI momentum and chop filter for regime filtering
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, capturing true trends while avoiding whipsaws.
# RSI adds momentum confirmation, and Choppiness Index filters ranging markets.
# Designed for low trade frequency (<25/year) to minimize fee drag in BTC/ETH.
# Works in bull (trend follow) and bear (mean revert in ranges) via regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, k=10))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close_1w, np.nan)
    kama[9] = close_1w[9]  # seed
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # 1d KAMA (10,30) for direction
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_1d = np.full_like(close, np.nan)
    kama_1d[9] = close[9]
    for i in range(10, len(close)):
        kama_1d[i] = kama_1d[i-1] + sc[i] * (close[i] - kama_1d[i-1])
    
    # 1d RSI(14) for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # 1d Choppiness Index(14) for regime
    atr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr[0] = high[0] - low[0]
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (hh14 - ll14)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(kama_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama_1d[i]
        price_below_kama = close[i] < kama_1d[i]
        
        # Weekly KAMA trend filter
        weekly_uptrend = close[i] > kama_1w_aligned[i]
        weekly_downtrend = close[i] < kama_1w_aligned[i]
        
        # RSI momentum: 40-60 for neutral, >60 for bullish, <40 for bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Chop regime: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
        chop_high = chop[i] > 61.8  # ranging
        chop_low = chop[i] < 38.2   # trending
        
        # Entry logic: In trending regime, follow KAMA+weekly+RSI
        # In ranging regime, mean revert at RSI extremes
        if chop_low and price_above_kama and weekly_uptrend and rsi_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        elif chop_low and price_below_kama and weekly_downtrend and rsi_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        elif chop_high and rsi[i] > 70 and position != -1:  # overbought -> short
            position = -1
            signals[i] = -0.25
        elif chop_high and rsi[i] < 30 and position != 1:  # oversold -> long
            position = 1
            signals[i] = 0.25
        # Exit: opposite conditions
        elif position == 1 and (chop_high and rsi[i] > 70 or not price_above_kama or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (chop_high and rsi[i] < 30 or not price_below_kama or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
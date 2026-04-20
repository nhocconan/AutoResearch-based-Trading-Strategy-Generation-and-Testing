#!/usr/bin/env python3
# 12h_1d_KAMA_Trend_Volume_Strategy
# Hypothesis: KAMA on 1d captures adaptive trend direction; combined with volume confirmation and RSI filter on 12h,
# it provides robust trend-following signals in both bull and bear markets. KAMA adapts to volatility,
# reducing whipsaws in choppy markets. Volume ensures institutional participation. RSI avoids overextended entries.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "12h_1d_KAMA_Trend_Volume_Strategy"
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
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average) on 1d close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(df_1d['close'], n=10))
    volatility = np.sum(np.abs(np.diff(df_1d['close'])), axis=1)
    # Use pandas for rolling sum to handle alignment
    volatility_series = pd.Series(np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0])))
    volatility_sum = volatility_series.rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.full_like(df_1d['close'], np.nan)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI (14) on 12h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (alpha = 1/14)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.5x 20-period average on 12h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure KAMA and RSI are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA(1d) + RSI < 60 (not overbought) + volume confirmation
            if close[i] > kama_aligned[i] and rsi[i] < 60 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA(1d) + RSI > 40 (not oversold) + volume confirmation
            elif close[i] < kama_aligned[i] and rsi[i] > 40 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price < KAMA(1d) or RSI > 70 (overbought)
            if close[i] < kama_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price > KAMA(1d) or RSI < 30 (oversold)
            if close[i] > kama_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
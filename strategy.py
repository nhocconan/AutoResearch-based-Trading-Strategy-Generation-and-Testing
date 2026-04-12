#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_Trend_Combined_v1
Hypothesis: Combine KAMA (trend-following) and RSI (mean-reversion) with 1d trend filter.
In trending markets (price > EMA200), use KAMA crossover for momentum entries.
In ranging markets (price near EMA200), use RSI extremes for mean reversion.
Uses volume confirmation to filter false signals. Targets 20-30 trades/year for low friction.
Works in bull (follow KAMA trend) and bear (RSI mean reversion at extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Trend_Combined_v1"
timeframe = "4h"
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
    
    # Daily data for trend filter and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA200 for trend regime filter
    daily_close = df_1d['close'].values
    ema200 = np.full(len(daily_close), np.nan)
    if len(daily_close) >= 200:
        alpha = 2 / (200 + 1)
        ema200[0] = daily_close[0]
        for i in range(1, len(daily_close)):
            ema200[i] = alpha * daily_close[i] + (1 - alpha) * ema200[i-1]
    ema200_4h = align_htf_to_ltf(prices, df_1d, ema200)
    
    # KAMA on 4h (trend following component)
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # simplified
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI on 4h (mean reversion component)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema200_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: trending vs ranging
        trending = close[i] > ema200_4h[i] * 1.02  # 2% above EMA200 = strong uptrend
        ranging = np.abs(close[i] - ema200_4h[i]) < ema200_4h[i] * 0.01  # within 1% of EMA200
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i] * 1.5
        
        # Signals based on regime
        if trending:
            # Trend following: KAMA crossover
            long_entry = close[i] > kama[i] and vol_confirm and close[i-1] <= kama[i-1]
            short_entry = close[i] < kama[i] and vol_confirm and close[i-1] >= kama[i-1]
        elif ranging:
            # Mean reversion: RSI extremes
            long_entry = rsi[i] < 30 and vol_confirm
            short_entry = rsi[i] > 70 and vol_confirm
        else:
            # Transition period: no entries
            long_entry = False
            short_entry = False
        
        # Exit conditions
        long_exit = (trending and close[i] < kama[i]) or (ranging and rsi[i] > 50)
        short_exit = (trending and close[i] > kama[i]) or (ranging and rsi[i] < 50)
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
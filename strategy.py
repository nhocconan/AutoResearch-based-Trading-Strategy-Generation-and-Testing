#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_v1
Hypothesis: Use daily KAMA direction as primary trend filter on 4H timeframe.
Enter long when 4H price crosses above KAMA and daily trend is up; short when price crosses below KAMA and daily trend is down.
Use RSI(14) to avoid overbought/oversold extremes (RSI<70 for long, RSI>30 for short).
Target 20-40 trades per year to minimize fee drag. Works in bull (follow trend) and bear (fade extremes in downtrend via RSI filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    def kama(close, length=10, fast=2, slow=30):
        if len(close) < length:
            return np.full(len(close), np.nan)
        dir = np.abs(close - np.roll(close, length))
        vol = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) > 1 else 0
        er = np.where(vol != 0, dir / vol, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama = np.full(len(close), np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    daily_close = df_1d['close'].values
    kama_daily = kama(daily_close, 10, 2, 30)
    kama_daily_aligned = align_htf_to_ltf(prices, df_1d, kama_daily)
    
    # Calculate RSI(14) on 4H
    def rsi(close, length=14):
        if len(close) < length + 1:
            return np.full(len(close), np.nan)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[:length])
        avg_loss[length] = np.mean(loss[:length])
        for i in range(length + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi_vals = np.zeros_like(close)
        rsi_vals[:] = 100 - (100 / (1 + rs))
        rsi_vals[:length] = np.nan
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if np.isnan(rsi_vals[i]) or np.isnan(kama_daily_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price relative to daily KAMA
        price_above_kama = close[i] > kama_daily_aligned[i]
        price_below_kama = close[i] < kama_daily_aligned[i]
        
        # RSI filters to avoid extremes
        rsi_not_overbought = rsi_vals[i] < 70
        rsi_not_oversold = rsi_vals[i] > 30
        
        # Entry logic
        long_entry = price_above_kama and rsi_not_overbought
        short_entry = price_below_kama and rsi_not_oversold
        
        # Exit logic: reverse signal
        long_exit = price_below_kama
        short_exit = price_above_kama
        
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
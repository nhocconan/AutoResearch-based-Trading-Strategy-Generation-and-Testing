#!/usr/bin/env python3
"""
12h_1w_KAMA_RSI_Trend_v1
Hypothesis: Use weekly trend (price above/below 50-week EMA) to filter 12H KAMA direction signals.
Add RSI(14) < 30 for long and > 70 for short to avoid chasing extremes.
KAMA adapts to market noise, reducing whipsaws in ranging markets.
Targets 20-30 trades per year to minimize fee drag. Works in bull (follow trend) and bear (fade extremes in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_KAMA_RSI_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema50 = np.full(len(weekly_close), np.nan)
    if len(weekly_close) >= 50:
        alpha = 2 / (50 + 1)
        ema50[0] = weekly_close[0]
        for i in range(1, len(weekly_close)):
            ema50[i] = alpha * weekly_close[i] + (1 - alpha) * ema50[i-1]
    
    # Align EMA50 to 12h
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Calculate KAMA on 12h close
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
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate RSI(14)
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
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: weekly close above/below EMA50
        weekly_close_price = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_price)
        trend_up = weekly_close_aligned[i] > ema50_aligned[i]
        
        # KAMA direction
        kama_up = kama_vals[i] > kama_vals[i-1]
        kama_down = kama_vals[i] < kama_vals[i-1]
        
        # RSI extremes
        rsi_oversold = rsi_vals[i] < 30
        rsi_overbought = rsi_vals[i] > 70
        
        # Entry logic
        long_entry = kama_up and rsi_oversold and trend_up
        short_entry = kama_down and rsi_overbought and not trend_up
        
        # Exit logic: reverse signal or RSI normalization
        long_exit = not kama_up or rsi_vals[i] > 50
        short_exit = not kama_down or rsi_vals[i] < 50
        
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
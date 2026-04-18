#!/usr/bin/env python3
"""
4h_1d_LongOnly_Trend_Follower
Hypothesis: Long-only strategy that buys on pullbacks to the 1d EMA21 when 4h momentum is positive (RSI>50) and price is above 1d EMA200 (bullish regime). 
Exit when price closes below 4h EMA13. Designed to avoid bear markets by staying flat, reducing whipsaw losses. 
Targets 15-25 trades/year with low turnover to minimize fee drag. Works in bull markets via trend continuation and avoids bear markets via regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA21 (for pullback entry)
    ema21_1d = np.full(len(close_1d), np.nan)
    k21 = 2 / (21 + 1)
    for i in range(21, len(close_1d)):
        if i == 21:
            ema21_1d[i] = np.mean(close_1d[i-21+1:i+1])
        else:
            ema21_1d[i] = close_1d[i] * k21 + ema21_1d[i-1] * (1 - k21)
    
    # 1d EMA200 (bullish regime filter)
    ema200_1d = np.full(len(close_1d), np.nan)
    k200 = 2 / (200 + 1)
    for i in range(200, len(close_1d)):
        if i == 200:
            ema200_1d[i] = np.mean(close_1d[i-200+1:i+1])
        else:
            ema200_1d[i] = close_1d[i] * k200 + ema200_1d[i-1] * (1 - k200)
    
    # Align 1d indicators to 4h
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # --- 4h Indicators (LTF) ---
    # 4h EMA13 (exit signal)
    ema13 = np.full(n, np.nan)
    k13 = 2 / (13 + 1)
    for i in range(13, n):
        if i == 13:
            ema13[i] = np.mean(close[i-13+1:i+1])
        else:
            ema13[i] = close[i] * k13 + ema13[i-1] * (1 - k13)
    
    # 4h RSI14 (momentum filter)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[i-14+1:i+1])
            avg_loss[i] = np.mean(loss[i-14+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 200  # Wait for 1d EMA200 to be valid
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(ema13[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price pulls back to near 1d EMA21, 4h momentum positive, bullish regime, volume confirmation
            # Allow 0.5% tolerance above EMA21 to account for discrete 4h bars
            if (close[i] >= ema21_1d_aligned[i] * 0.995 and  # Within 0.5% below EMA21 (pullback)
                close[i] <= ema21_1d_aligned[i] * 1.005 and  # Within 0.5% above EMA21
                rsi[i] > 50 and                             # 4h bullish momentum
                close[i] > ema200_1d_aligned[i] and         # Bullish regime: above 1d EMA200
                vol_filter[i]):                             # Volume confirmation
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit: price closes below 4h EMA13 (trend weakness)
            if close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "4h_1d_LongOnly_Trend_Follower"
timeframe = "4h"
leverage = 1.0
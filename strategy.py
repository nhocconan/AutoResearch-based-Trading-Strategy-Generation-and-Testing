#!/usr/bin/env python3
"""
1d_1w_RSI_MeanReversion_v1
Hypothesis: On daily timeframe, RSI extremes (overbought/oversold) combined with weekly trend filter (price above/below weekly SMA50) provide mean-reversion entries that work in both bull and bear markets. Weekly trend filter prevents counter-trend trades during strong moves, reducing whipsaws. Targets 20-50 total trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_MeanReversion_v1"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly SMA50 for trend
    weekly_close = df_1w['close'].values
    weekly_sma50 = np.full(len(weekly_close), np.nan)
    for i in range(50, len(weekly_close)):
        weekly_sma50[i] = np.mean(weekly_close[i-50:i])
    
    # Daily RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly trend to daily
    weekly_sma50_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma50)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    weekly_trend_up = weekly_close_aligned > weekly_sma50_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(rsi[i]) or np.isnan(weekly_sma50_aligned[i]) or 
            np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry logic: counter-trend to weekly trend
        long_entry = rsi_oversold and weekly_trend_up[i]
        short_entry = rsi_overbought and not weekly_trend_up[i]
        
        # Exit conditions: RSI returns to neutral zone
        long_exit = rsi[i] > 50
        short_exit = rsi[i] < 50
        
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
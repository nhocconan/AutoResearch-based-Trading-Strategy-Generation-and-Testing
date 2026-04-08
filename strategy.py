#!/usr/bin/env python3
# daily_ema_rsi_weekly_v1
# Hypothesis: 1-day EMA200 trend filter with RSI(14) mean reversion and weekly trend confirmation.
# Long when: price > EMA200 AND RSI < 30 AND weekly close > weekly EMA50
# Short when: price < EMA200 AND RSI > 70 AND weekly close < weekly EMA50
# Exit when: price crosses EMA200 OR RSI crosses back above 50 (long) or below 50 (short)
# Uses daily for entry/exit and weekly for trend filter to avoid counter-trend trades.
# Target: 10-25 trades/year to minimize fee drag while capturing high-probability mean reversions.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "daily_ema_rsi_weekly_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate daily EMA200 for trend filter
    ema200 = np.full(n, np.nan)
    ema200[199] = np.mean(close[:200])
    for i in range(200, n):
        ema200[i] = close[i] * 0.00995 + ema200[i-1] * 0.99005
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (gain[i] * 1/14 + avg_gain[i-1] * 13/14)
        avg_loss[i] = (loss[i] * 1/14 + avg_loss[i-1] * 13/14)
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA50
    ema_1w_50 = np.zeros(len(close_1w))
    ema_1w_50[:] = np.nan
    ema_1w_50[49] = np.mean(close_1w[:50])
    for i in range(50, len(close_1w)):
        ema_1w_50[i] = close_1w[i] * 0.0377 + ema_1w_50[i-1] * 0.9623
    
    # Align both weekly close and weekly EMA50 to daily timeframe
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        price = close[i]
        ema = ema200[i]
        rsi_val = rsi[i]
        weekly_close = close_1w_aligned[i]
        weekly_ema = ema_1w_50_aligned[i]
        
        if np.isnan(ema) or np.isnan(rsi_val) or np.isnan(weekly_close) or np.isnan(weekly_ema):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price below EMA200 OR RSI > 50
            if price < ema or rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price above EMA200 OR RSI < 50
            if price > ema or rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: long setup
            if price > ema and rsi_val < 30 and weekly_close > weekly_ema:
                position = 1
                signals[i] = 0.25
            # Entry: short setup
            elif price < ema and rsi_val > 70 and weekly_close < weekly_ema:
                position = -1
                signals[i] = -0.25
    
    return signals
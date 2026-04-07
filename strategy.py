#!/usr/bin/env python3
"""
1d_kama_rsi_chop_v1
Hypothesis: On daily timeframe, use KAMA for trend direction, RSI for momentum, and Choppiness Index for regime filtering. Enter long when KAMA slope is positive, RSI < 30 (oversold), and market is choppy (CHOP > 61.8); enter short when KAMA slope is negative, RSI > 70 (overbought), and market is choppy. Exit when RSI reaches opposite extreme or trend changes. This mean-reversion strategy works in ranging markets (2025-2026) and avoids trends via chop filter, reducing false signals.
"""

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
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend and chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # KAMA on weekly data
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if not np.isnan(sc[i]):
            kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
        else:
            kama_1w[i] = kama_1w[i-1]
    
    # RSI on weekly data (14-period)
    delta = np.diff(close_1w)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Choppiness Index on weekly data (14-period)
    atr_1w = []
    for i in range(len(df_1w)):
        if i == 0:
            atr_1w.append(np.nan)
        else:
            tr = max(
                df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
                np.abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
                np.abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
            )
            atr_1w.append(tr)
    atr_1w = np.array(atr_1w)
    # True range sum over 14 periods
    tr_sum = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    # Max/min range over 14 periods
    max_high = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    range_maxmin = max_high - min_low
    # Chop calculation
    chop_1w = 100 * np.log10(tr_sum / range_maxmin) / np.log10(14)
    
    # Align indicators to daily timeframe
    kama_1w_1d = align_htf_to_ltf(prices, df_1w, kama_1w)
    rsi_1w_1d = align_htf_to_ltf(prices, df_1w, rsi_1w)
    chop_1w_1d = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama_1w_1d[i]) or np.isnan(rsi_1w_1d[i]) or np.isnan(chop_1w_1d[i])):
            signals[i] = 0.0
            continue
        
        # KAMA slope (3-period change)
        if i >= 3:
            kama_slope = kama_1w_1d[i] - kama_1w_1d[i-3]
        else:
            kama_slope = 0
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if RSI reaches overbought
            if rsi_1w_1d[i] > 70:
                exit_long = True
            # Exit if trend turns down (KAMA slope negative)
            elif kama_slope < 0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if RSI reaches oversold
            if rsi_1w_1d[i] < 30:
                exit_short = True
            # Exit if trend turns up (KAMA slope positive)
            elif kama_slope > 0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # KAMA slope positive (uptrend), RSI oversold, choppy market
            if kama_slope > 0 and rsi_1w_1d[i] < 30 and chop_1w_1d[i] > 61.8:
                long_entry = True
            
            # Short entry conditions
            short_entry = False
            # KAMA slope negative (downtrend), RSI overbought, choppy market
            if kama_slope < 0 and rsi_1w_1d[i] > 70 and chop_1w_1d[i] > 61.8:
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals
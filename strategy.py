#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Chop_Filter_v1
Hypothesis: Daily KAMA direction with RSI momentum and weekly chop filter.
Works in bull markets via KAMA trend following and in bear markets via RSI mean reversion
when choppy. Weekly chop filter avoids whipsaws in strong trends. Target 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_Filter_v1"
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
    
    # Load weekly data ONCE before loop for chop filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate KAMA (20 period) on daily
    close_d = df_1w['close'].values  # Using weekly close for efficiency
    change = np.abs(np.diff(close_d, prepend=close_d[0]))
    direction = np.abs(np.diff(close_d, k=10, prepend=close_d[:10]))
    er = np.where(change != 0, direction / change, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_d)
    kama[0] = close_d[0]
    for i in range(1, len(close_d)):
        kama[i] = kama[i-1] + sc[i] * (close_d[i] - kama[i-1])
    kama_1d = align_htf_to_ltf(prices, df_1w, kama)
    
    # Calculate RSI (14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate Choppiness Index (14) on weekly
    atr_1w = []
    for i in range(len(df_1w)):
        if i == 0:
            atr_1w.append(0)
        else:
            tr = max(
                df_1w['high'].iloc[i] - df_1w['low'].iloc[i],
                abs(df_1w['high'].iloc[i] - df_1w['close'].iloc[i-1]),
                abs(df_1w['low'].iloc[i] - df_1w['close'].iloc[i-1])
            )
            atr_1w.append(tr)
    atr_1w = np.array(atr_1w)
    sum_atr = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum()
    hh = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max()
    ll = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(sum_atr / (hh - ll)) / np.log10(14)
    chop = chop.fillna(50).values
    chop_1d = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_1d[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_1d[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA trend: price above/below KAMA
        above_kama = close[i] > kama_1d[i]
        below_kama = close[i] < kama_1d[i]
        
        # RSI momentum: oversold/overbought
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Chop filter: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending (trend follow)
        chop_value = chop_1d[i]
        ranging = chop_value > 61.8
        trending = chop_value < 38.2
        
        # Entry conditions
        long_entry = (above_kama and rsi_oversold and ranging) or (above_kama and trending)
        short_entry = (below_kama and rsi_overbought and ranging) or (below_kama and trending)
        
        # Exit conditions: opposite signal or extreme RSI
        long_exit = (below_kama) or (rsi[i] > 70)
        short_exit = (above_kama) or (rsi[i] < 30)
        
        # Priority: entry > exit > hold
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
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
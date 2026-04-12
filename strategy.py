#!/usr/bin/env python3
"""
1d_1w_KAMA_RSI_Chop_Filter_v2
Hypothesis: Daily KAMA trend + RSI mean reversion + weekly chop filter.
KAMA adapts to market efficiency, reducing whipsaw in chop. RSI(14) < 30 for long, > 70 for short.
Weekly chop > 50 filters out trend days, favoring mean reversion in ranging markets.
Works in bull/bear by only taking mean-reversion entries when market is choppy (range-bound).
Target: 10-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_KAMA_RSI_Chop_Filter_v2"
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
    
    # Calculate KAMA (adaptive moving average) on daily closes
    close_series = pd.Series(close)
    # Efficiency ratio: price change / volatility
    change = abs(close - close.shift(10))
    volatility = abs(close - close.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate weekly chopiness index
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(abs(h - c_prev), abs(l - c_prev)))
    
    tr = true_range(df_1w['high'].values, df_1w['low'].values, 
                    np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].values[:-1]]))
    atr1 = pd.Series(tr).rolling(window=1, min_periods=1).sum()
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(atr14 / atr1) / np.log10(14)
    chop = chop.fillna(50)  # neutral when insufficient data
    
    # Align chop to daily
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop.values)
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Chop filter: only trade when market is choppy (range-bound)
        choppy = chop_aligned[i] > 50
        
        # RSI mean reversion signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # KAMA trend filter: price above/below KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # Entry conditions: RSI extreme + choppy + KAMA alignment
        long_entry = rsi_oversold and choppy and above_kama
        short_entry = rsi_overbought and choppy and below_kama
        
        # Exit conditions: RSI returns to neutral or trend change
        long_exit = (rsi[i] > 50) or (close[i] < kama[i])
        short_exit = (rsi[i] < 50) or (close[i] > kama[i])
        
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
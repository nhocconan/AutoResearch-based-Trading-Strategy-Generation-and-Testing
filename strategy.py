#!/usr/bin/env python3
"""
1d_1w_Breakout_Trend_Volume_Strategy
Hypothesis: Daily breakouts of 10-day high/low filtered by weekly EMA trend and volume expansion.
Captures strong trending moves in both bull and bear markets. Low trade frequency to avoid fee drag.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # True Range and ATR for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily breakout levels: 10-day high/low
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: current volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume > (vol_ma_20 * 1.3)
    
    # Weekly EMA trend filter (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        ema21_1w = np.full(len(prices), np.nan)
    else:
        close_1w = df_1w['close'].values
        ema21_1w_raw = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
        ema21_1w = align_htf_to_ltf(prices, df_1w, ema21_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # warmup period
        # Skip if any required data is not ready
        if (np.isnan(high_10[i]) or np.isnan(low_10[i]) or 
            np.isnan(ema21_1w[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long signal: break above 10-day high with volume expansion and weekly uptrend
        long_signal = (high[i] > high_10[i] and 
                      volume_expansion[i] and 
                      close[i] > ema21_1w[i])
        
        # Short signal: break below 10-day low with volume expansion and weekly downtrend
        short_signal = (low[i] < low_10[i] and 
                       volume_expansion[i] and 
                       close[i] < ema21_1w[i])
        
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Breakout_Trend_Volume_Strategy"
timeframe = "1d"
leverage = 1.0
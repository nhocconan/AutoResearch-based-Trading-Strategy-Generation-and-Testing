#!/usr/bin/env python3
# 6h_RSI_Volume_Trend_Filter
# Hypothesis: Combines RSI(14) with volume confirmation and trend filter from 12h EMA.
# Long when RSI crosses above 50, volume > 1.5x 20-bar average, and close > 12h EMA50.
# Short when RSI crosses below 50, volume > 1.5x 20-bar average, and close < 12h EMA50.
# Designed to capture momentum shifts with volume confirmation and trend alignment.
# Works in bull/bear by requiring trend alignment, reducing whipsaws.

name = "6h_RSI_Volume_Trend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA50 on 12h
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # RSI(14) on 6h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI crosses above 50, volume confirmation, and above 12h EMA50
            if (rsi[i] > 50 and 
                rsi[i-1] <= 50 and 
                volume_filter[i] and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50, volume confirmation, and below 12h EMA50
            elif (rsi[i] < 50 and 
                  rsi[i-1] >= 50 and 
                  volume_filter[i] and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI crosses below 50 (momentum shift)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI crosses above 50 (momentum shift)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
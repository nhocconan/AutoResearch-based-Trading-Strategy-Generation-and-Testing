#!/usr/bin/env python3
"""
12h_RSI_Overbought_Oversold_With_Trend_Filter_v1
Hypothesis: In ranging markets (2025-2026), RSI extremes on 12h timeframe provide mean reversion opportunities when filtered by 1d trend. Long when RSI < 30 and 1d close > 200 EMA (bullish bias). Short when RSI > 70 and 1d close < 200 EMA (bearish bias). Exit when RSI returns to neutral (40-60). Works in both bull/bear by following higher timeframe trend while exploiting mean reversion at extremes.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (200 EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(200) on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Load 12h data for RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.values
    # Align to 12h timeframe (no additional delay needed for RSI)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: RSI oversold with bullish trend bias
            if (rsi_12h_aligned[i] < 30 and 
                price > ema_200_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: RSI overbought with bearish trend bias
            elif (rsi_12h_aligned[i] > 70 and 
                  price < ema_200_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or trend turns bearish
            if (rsi_12h_aligned[i] >= 40 and rsi_12h_aligned[i] <= 60) or price < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or trend turns bullish
            if (rsi_12h_aligned[i] >= 40 and rsi_12h_aligned[i] <= 60) or price > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_Overbought_Oversold_With_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
1d_1w_RSI_Overbought_Oversold_With_Trend_Filter
Hypothesis: Use weekly RSI extremes with 1d trend filter for mean reversion. Long when weekly RSI < 30 and 1d price > EMA50. Short when weekly RSI > 70 and 1d price < EMA50. Exit when RSI returns to neutral (40-60). Weekly RSI avoids noise, trend filter prevents fighting the trend. Designed for 1d timeframe to capture multi-week mean reversion with ~10-25 trades/year. Works in bull markets by buying dips in uptrend and in bear markets by selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1w)
    avg_loss = np.zeros_like(close_1w)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w[:13] = np.nan  # Not enough data
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # 1d EMA50 for trend filter
    close_s = prices['close']
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(ema_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: weekly RSI oversold (<30) and price above EMA50 (uptrend)
            if rsi_1w_aligned[i] < 30 and price > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: weekly RSI overbought (>70) and price below EMA50 (downtrend)
            elif rsi_1w_aligned[i] > 70 and price < ema_50[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly RSI returns to neutral (>40) or turns bearish
            if rsi_1w_aligned[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly RSI returns to neutral (<60) or turns bullish
            if rsi_1w_aligned[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_RSI_Overbought_Oversold_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0
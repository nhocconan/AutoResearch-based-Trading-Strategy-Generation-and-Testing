#!/usr/bin/env python3
# [24955] 6h_1d_rsi_pullback_trend_v2
# Hypothesis: 6-hour RSI pullback in trend direction from 1-day trend filter.
# Long when RSI(14) < 30 (oversold) and price > 1-day EMA(50) (uptrend).
# Short when RSI(14) > 70 (overbought) and price < 1-day EMA(50) (downtrend).
# Exit when RSI returns to 50 (mean reversion) or opposite RSI extreme.
# Uses 1-day EMA for trend bias to avoid counter-trend trades in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_pullback_trend_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * (2/51) + ema_50_1d[i-1] * (49/51)
    
    # Calculate RSI(14) on 6-hour data
    rsi = np.full(n, np.nan)
    if n >= 15:
        # Calculate price changes
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Average gain and loss
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        
        # Initial average
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        
        # Wilder's smoothing
        for i in range(15, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        # Calculate RSI
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100
        rsi[:14] = np.nan
    
    # Align 1-day EMA(50) to 6-hour timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(15, n):  # Start after RSI warmup
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: RSI returns to 50 or goes overbought (>70)
            if rsi_val >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: RSI returns to 50 or goes oversold (<30)
            if rsi_val <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: RSI oversold (<30) and price above 1-day EMA(50) (uptrend)
            if rsi_val < 30 and price > ema_trend:
                position = 1
                signals[i] = 0.25
            # Enter short: RSI overbought (>70) and price below 1-day EMA(50) (downtrend)
            elif rsi_val > 70 and price < ema_trend:
                position = -1
                signals[i] = -0.25
    
    return signals
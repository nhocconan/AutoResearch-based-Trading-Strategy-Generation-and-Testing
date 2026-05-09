#!/usr/bin/env python3
# 4h_RSI_MeanReversion_TrendFilter
# Hypothesis: Mean reversion using RSI(14) with trend filter from 1d EMA50. Buy when RSI<30 and price>EMA50 (bullish mean reversion), sell when RSI>70 and price<EMA50 (bearish mean reversion). Works in both bull and bear markets by only taking mean reversion trades in the direction of the higher timeframe trend, reducing false signals. Target: 20-40 trades/year.

name = "4h_RSI_MeanReversion_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA(50) with proper initialization
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close, np.nan)
    rsi = np.full_like(close, np.nan)
    valid = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss != 0)
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi[valid] = 100 - (100 / (1 + rs[valid]))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need RSI and EMA warmed up
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold (<30) AND price above 1d EMA50 (bullish trend)
            if rsi[i] < 30 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) AND price below 1d EMA50 (bearish trend)
            elif rsi[i] > 70 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought (>70) or trend turns bearish
            if rsi[i] > 70 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold (<30) or trend turns bullish
            if rsi[i] < 30 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 1d_RSI_Extreme_Trend_Weekly
# Hypothesis: On daily timeframe, enter long when RSI(14) < 30 and price > weekly 50 EMA, short when RSI(14) > 70 and price < weekly 50 EMA.
# Uses weekly trend filter to avoid counter-trend trades in strong trends. RSI extremes provide mean reversion entries within the trend.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

name = "1d_RSI_Extreme_Trend_Weekly"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 2 + ema_50_1w[i-1] * 49) / 50
    
    # Align weekly EMA to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(close) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close)):
            avg_gain[i] = (gain[i] + avg_gain[i-1] * 13) / 14
            avg_loss[i] = (loss[i] + avg_loss[i-1] * 13) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold AND price above weekly EMA50 (uptrend)
            if rsi[i] < 30 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought AND price below weekly EMA50 (downtrend)
            elif rsi[i] > 70 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought OR price below weekly EMA50 (trend change)
            if rsi[i] > 70 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold OR price above weekly EMA50 (trend change)
            if rsi[i] < 30 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
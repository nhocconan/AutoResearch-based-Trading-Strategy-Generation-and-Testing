#!/usr/bin/env python3
"""
1d_1w_WeeklyEMA34_RSI25_75
Hypothesis: Trade long when price crosses above daily close while above weekly EMA34 and RSI < 25 (oversold), short when price crosses below daily close while below weekly EMA34 and RSI > 75 (overbought). Uses weekly trend filter to avoid counter-trend trades and RSI extremes for mean reversion within trend. Position size 0.25 targeting ~15-20 trades/year to minimize fee drag. Works in bull/bear by trading with trend and mean reversion at extremes.
"""

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
    
    # Get daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Daily RSI (14-period)
    delta = np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi_period = 14
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period - 1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period - 1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Weekly EMA (34-period)
    close_1w = df_1w['close'].values
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align daily RSI to 1d timeframe (already aligned)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Align weekly EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, rsi_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA and RSI oversold (<25)
            if close[i] > ema_1w_aligned[i] and rsi_1d_aligned[i] < 25:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA and RSI overbought (>75)
            elif close[i] < ema_1w_aligned[i] and rsi_1d_aligned[i] > 75:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly EMA or RSI overbought (>70)
            if close[i] < ema_1w_aligned[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly EMA or RSI oversold (<30)
            if close[i] > ema_1w_aligned[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_WeeklyEMA34_RSI25_75"
timeframe = "1d"
leverage = 1.0
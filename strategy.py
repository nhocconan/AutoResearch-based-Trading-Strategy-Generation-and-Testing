#!/usr/bin/env python3
"""
12h_RSI20_1wTrend_Pullback
Hypothesis: On 12h timeframe, buy pullbacks in weekly uptrend (price > weekly EMA50) when RSI(14) < 20, and sell rallies in weekly downtrend (price < weekly EMA50) when RSI(14) > 80. Uses weekly EMA50 as trend filter and RSI extremes for mean-reversion entries. Designed for 12-37 trades per year on 12h timeframe, works in bull via buying dips in uptrend, bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for EMA filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])  # first average of gains
    avg_loss[14] = np.mean(loss[1:15])  # first average of losses
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RSI calculation
    start_idx = 14
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi[i]
        ema_50_val = ema_50_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: weekly uptrend (price > weekly EMA50) AND RSI < 20 (oversold)
            if price > ema_50_val and rsi_val < 20:
                signals[i] = size
                position = 1
            # Short: weekly downtrend (price < weekly EMA50) AND RSI > 80 (overbought)
            elif price < ema_50_val and rsi_val > 80:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or trend change (price < weekly EMA50)
            if rsi_val > 60 or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or trend change (price > weekly EMA50)
            if rsi_val < 40 or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_RSI20_1wTrend_Pullback"
timeframe = "12h"
leverage = 1.0
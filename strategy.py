#!/usr/bin/env python3
"""
12h_1d_RSI_Momentum_Trend_v1
Hypothesis: 12h timeframe with 1d RSI momentum and 1d EMA trend filter.
Long when 1d RSI > 60 and price > 1d EMA, short when 1d RSI < 40 and price < 1d EMA.
Uses 1d data for signals to avoid overtrading (target ~15-25 trades/year).
Works in bull markets via RSI momentum and in bear via mean-reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_Momentum_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI (14 period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan  # Not enough data
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d EMA (21 period) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Momentum conditions: RSI > 60 for long, < 40 for short
        rsi_high = rsi_1d_aligned[i] > 60
        rsi_low = rsi_1d_aligned[i] < 40
        
        # Trend filter: price above/below 1d EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Entry conditions
        long_entry = rsi_high and above_ema
        short_entry = rsi_low and below_ema
        
        # Exit conditions: RSI returns to neutral zone or trend reversal
        long_exit = (rsi_1d_aligned[i] < 50) or (close[i] < ema_1d_aligned[i])
        short_exit = (rsi_1d_aligned[i] > 50) or (close[i] > ema_1d_aligned[i])
        
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
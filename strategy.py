#!/usr/bin/env python3
"""
1h_RSI_Pullback_4hTrend
Hypothesis: In 4h trend context (EMA50), enter on RSI(14) pullbacks (long when RSI<40, short when RSI>60) at 1h timeframe.
Volume confirmation (>1.5x 20-bar avg) filters false signals. Session filter (08-20 UTC) reduces noise.
Designed for low trade frequency (target: 15-37/year) with defined risk via trend-following exits.
Works in bull/bear by following 4h trend while using mean-reversion entries.
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
    volume = prices['volume'].values
    
    # 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = np.full(len(close_4h), np.nan)
    for i in range(50, len(close_4h)):
        if i == 50:
            ema50_4h[i] = np.mean(close_4h[0:51])
        else:
            k = 2 / (50 + 1)
            ema50_4h[i] = close_4h[i] * k + ema50_4h[i-1] * (1 - k)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # RSI(14) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-bar average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 40 (oversold) in 4h uptrend with volume
            if (rsi[i] < 40 and close[i] > ema50_4h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 60 (overbought) in 4h downtrend with volume
            elif (rsi[i] > 60 and close[i] < ema50_4h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 60 (overbought) or 4h trend turns down
            if (rsi[i] > 60 or close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI < 40 (oversold) or 4h trend turns up
            if (rsi[i] < 40 or close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Pullback_4hTrend"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_RSI_MeanReversion_With_Trend_Filter
Hypothesis: Mean reversion using RSI(14) with daily EMA34 trend filter and volume confirmation.
Works in both bull and bear markets by only taking mean-reversion trades in the direction of the higher timeframe trend.
Designed for low trade frequency (target: 20-50/year) with strict entry conditions to avoid overtrading.
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
    
    # Calculate daily EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = close_1d[i] * alpha + ema34_1d[i-1] * (1 - alpha)
    
    # Align daily EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate RSI(14)
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
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) with volume spike and daily uptrend
            if (rsi[i] < 30 and vol_spike[i] and close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) with volume spike and daily downtrend
            elif (rsi[i] > 70 and vol_spike[i] and close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or trend turns down
            if (rsi[i] > 70 or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or trend turns up
            if (rsi[i] < 30 or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_MeanReversion_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0
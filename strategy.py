#!/usr/bin/env python3
"""
1h_4h1d_Trend_Follow_With_Volume
Hypothesis: Use 4h EMA trend and 1d RSI filter for direction, with volume confirmation on 1h for entry.
Aims to capture trending moves while avoiding chop, targeting 15-30 trades/year.
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
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34)
    ema34_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 34:
        ema34_4h[33] = np.mean(close_4h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_4h)):
            ema34_4h[i] = close_4h[i] * alpha + ema34_4h[i-1] * (1 - alpha)
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align HTF indicators
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend, 1d RSI not overbought, volume spike
            if (close[i] > ema34_4h_aligned[i] and 
                rsi_1d_aligned[i] < 70 and vol_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, 1d RSI not oversold, volume spike
            elif (close[i] < ema34_4h_aligned[i] and 
                  rsi_1d_aligned[i] > 30 and vol_spike[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend turns down or RSI overbought
            if (close[i] < ema34_4h_aligned[i] or rsi_1d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend turns up or RSI oversold
            if (close[i] > ema34_4h_aligned[i] or rsi_1d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Trend_Follow_With_Volume"
timeframe = "1h"
leverage = 1.0
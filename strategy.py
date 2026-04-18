#!/usr/bin/env python3
"""
12h_Momentum_Reversal_Confluence
Hypothesis: Mean reversion at daily extremes with momentum confirmation works in both bull and bear markets. 
Uses daily RSI extremes (>70/<30) combined with 12-hour momentum reversal signals and volume confirmation.
Designed for low trade frequency (15-30 trades/year) with strong performance in ranging and trending markets.
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
    
    # Calculate daily RSI(14) for overbought/oversold signals
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI with proper smoothing
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:15])
        avg_loss[13] = np.mean(loss[1:15])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 12h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 12-period RSI on 12h timeframe for momentum
    delta_12h = np.diff(close, prepend=close[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    
    avg_gain_12h = np.full(n, np.nan)
    avg_loss_12h = np.full(n, np.nan)
    
    if n >= 14:
        avg_gain_12h[13] = np.mean(gain_12h[1:15])
        avg_loss_12h[13] = np.mean(loss_12h[1:15])
        for i in range(14, n):
            avg_gain_12h[i] = (avg_gain_12h[i-1] * 13 + gain_12h[i]) / 14
            avg_loss_12h[i] = (avg_loss_12h[i-1] * 13 + loss_12h[i]) / 14
    
    rs_12h = np.divide(avg_gain_12h, avg_loss_12h, out=np.full_like(avg_gain_12h, np.nan), where=avg_loss_12h!=0)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily oversold + 12h momentum turning up + volume spike
            if (rsi_14_aligned[i] < 30 and rsi_12h[i] > 50 and 
                rsi_12h[i] > rsi_12h[i-1] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: daily overbought + 12h momentum turning down + volume spike
            elif (rsi_14_aligned[i] > 70 and rsi_12h[i] < 50 and 
                  rsi_12h[i] < rsi_12h[i-1] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: daily overbought or momentum deteriorates
            if (rsi_14_aligned[i] > 70 or rsi_12h[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: daily oversold or momentum improves
            if (rsi_14_aligned[i] < 30 or rsi_12h[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Momentum_Reversal_Confluence"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_RSI_Divergence_Trend_Filter
Hypothesis: RSI divergence with price trend filter. Bullish divergence (higher low in RSI, lower low in price) + price > EMA50 triggers long. Bearish divergence (lower high in RSI, higher high in price) + price < EMA50 triggers short. Uses 4h timeframe with 1d trend filter to avoid counter-trend trades. Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets by catching reversals at trend exhaustion points.
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
    
    # Calculate RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Calculate EMA50 for trend filter
    ema50 = np.full(n, np.nan)
    if n >= 50:
        ema50[49] = np.mean(close[0:50])
        alpha = 2 / (50 + 1)
        for i in range(50, n):
            ema50[i] = close[i] * alpha + ema50[i-1] * (1 - alpha)
    
    # Calculate daily EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[0:50])
        alpha_1d = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = close_1d[i] * alpha_1d + ema50_1d[i-1] * (1 - alpha_1d)
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema50[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for RSI divergence
        bullish_div = False
        bearish_div = False
        
        if i >= 5:  # Need at least 5 periods to check for divergence
            # Bullish divergence: RSI makes higher low, price makes lower low
            if (rsi[i] > rsi[i-3] and rsi[i-3] > rsi[i-6] and  # RSI higher low
                close[i] < close[i-3] and close[i-3] < close[i-6]):  # Price lower low
                bullish_div = True
            
            # Bearish divergence: RSI makes lower high, price makes higher high
            if (rsi[i] < rsi[i-3] and rsi[i-3] < rsi[i-6] and  # RSI lower high
                close[i] > close[i-3] and close[i-3] > close[i-6]):  # Price higher high
                bearish_div = True
        
        if position == 0:
            # Long: bullish RSI divergence + price above EMA50 + daily uptrend + volume spike
            if (bullish_div and close[i] > ema50[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence + price below EMA50 + daily downtrend + volume spike
            elif (bearish_div and close[i] < ema50[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish RSI divergence or price below EMA50
            if (bearish_div or close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish RSI divergence or price above EMA50
            if (bullish_div or close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Divergence_Trend_Filter"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h_RSI_Divergence_BullBear
Hypothesis: RSI divergence with price on 12h timeframe identifies exhaustion points in both bull and bear markets. 
Combined with volume confirmation and 1-day trend filter to avoid false signals in ranging markets.
Target: 15-30 trades/year on 12h timeframe with disciplined entry conditions.
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
    
    # RSI calculation (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    rsi = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # 1-day EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema34_1d = np.full(len(close_1d), np.nan)
    k = 2 / (34 + 1)
    for i in range(34, len(close_1d)):
        if i == 34:
            ema34_1d[i] = np.mean(close_1d[0:35])
        else:
            ema34_1d[i] = close_1d[i] * k + ema34_1d[i-1] * (1 - k)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5 x 12-period average
    vol_ma = np.full(n, np.nan)
    for i in range(12, n):
        vol_ma[i] = np.mean(volume[i-12:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check for RSI divergence
        bullish_div = False
        bearish_div = False
        
        if i >= 25:  # Need at least 25 bars to look back for divergence
            # Look for bullish divergence: price makes lower low, RSI makes higher low
            if low[i] < low[i-5] and low[i-5] < low[i-10] and low[i-10] < low[i-15] and low[i-15] < low[i-20]:
                if rsi[i] > rsi[i-5] and rsi[i-5] > rsi[i-10] and rsi[i-10] > rsi[i-15] and rsi[i-15] > rsi[i-20]:
                    bullish_div = True
            
            # Look for bearish divergence: price makes higher high, RSI makes lower high
            if high[i] > high[i-5] and high[i-5] > high[i-10] and high[i-10] > high[i-15] and high[i-15] > high[i-20]:
                if rsi[i] < rsi[i-5] and rsi[i-5] < rsi[i-10] and rsi[i-10] < rsi[i-15] and rsi[i-15] < rsi[i-20]:
                    bearish_div = True
        
        if position == 0:
            # Long: bullish RSI divergence with volume spike and 1-day uptrend
            if (bullish_div and vol_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish RSI divergence with volume spike and 1-day downtrend
            elif (bearish_div and vol_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish RSI divergence or 1-day trend turns down
            if (bearish_div or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish RSI divergence or 1-day trend turns up
            if (bullish_div or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_RSI_Divergence_BullBear"
timeframe = "12h"
leverage = 1.0
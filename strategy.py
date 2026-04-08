#!/usr/bin/env python3
"""
4h KAMA + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, providing a smooth trend line. RSI identifies overbought/oversold conditions, while the Choppiness Index filters for trending markets. This combination works in both bull and bear markets by aligning with adaptive trend and momentum extremes only when the market is trending (not choppy). Targets 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_kama_rsi_chop_filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend filter: EMA(50)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA (adaptive moving average) - ER = 10, FC = 2, 30
    change = np.abs(np.diff(close, k=10))  # 10-period change
    abs_change = np.abs(np.diff(close))    # 1-period change
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.where(np.convolve(abs_change, np.ones(10), mode='same')[10:] == 0, 1, np.convolve(abs_change, np.ones(10), mode='same')[10:])
    sc = (er * (2/(30+1) - 2/(2+1)) + 2/(2+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.convolve(tr, np.ones(14)/14, mode='same')
    max_high = np.zeros_like(close)
    min_low = np.zeros_like(close)
    for i in range(n):
        if i < 14:
            max_high[i] = np.max(high[:i+1])
            min_low[i] = np.min(low[:i+1])
        else:
            max_high[i] = np.max(high[i-13:i+1])
            min_low[i] = np.min(low[i-13:i+1])
    chop = np.where((max_high - min_low) != 0, 100 * np.log10(atr * 14 / (max_high - min_low)) / np.log10(14), 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA OR RSI overbought (>70) OR chop too high (>61.8)
            if (close[i] < kama[i] or 
                rsi[i] > 70 or 
                chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA OR RSI oversold (<30) OR chop too high (>61.8)
            if (close[i] > kama[i] or 
                rsi[i] < 30 or 
                chop[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price above KAMA, RSI oversold (<30), trending market (chop < 38.2), uptrend
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop[i] < 38.2 and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: price below KAMA, RSI overbought (>70), trending market (chop < 38.2), downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] < 38.2 and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
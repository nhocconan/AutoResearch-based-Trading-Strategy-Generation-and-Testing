#!/usr/bin/env python3
"""
12h_KAMA_RSI_Volume_Signal
Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) to detect trend direction, combined with RSI for momentum confirmation and volume surge for entry timing. This strategy aims to capture trend continuations with low frequency to minimize fee drag, working in both bull and bear markets by aligning with adaptive trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for volume confirmation and trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) on close prices
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume average for confirmation
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # Daily trend filter: price above/below 50 EMA
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers KAMA, RSI, and daily indicators
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_avg20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.8x 20-day average (tighter to reduce trades)
        vol_filter = vol_1d_current > 1.8 * vol_avg20_1d_aligned[i]
        
        # Trend filter: price vs KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # RSI conditions: not overbought/oversold
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Entry conditions
        if position == 0:
            # Long: price > KAMA + volume surge + RSI not overbought + above daily EMA
            if (above_kama and vol_filter and rsi_not_overbought and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < KAMA + volume surge + RSI not oversold + below daily EMA
            elif (below_kama and vol_filter and rsi_not_oversold and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal when price crosses KAMA
        elif position == 1:
            if close[i] < kama[i]:  # price crosses below KAMA = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > kama[i]:  # price crosses above KAMA = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Volume_Signal"
timeframe = "12h"
leverage = 1.0
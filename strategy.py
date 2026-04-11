#!/usr/bin/env python3
"""
12h_1d_RSI_Reversal_with_Volume_and_Trend_Filter
Hypothesis: Uses RSI extremes combined with volume spikes and 1-day trend filter on 12h timeframe.
Designed for fewer trades (target: 15-30/year) by requiring RSI <25 for long, >75 for short,
volume >2x 24-period average, and trend alignment via 1-day EMA50. Works in both bull and bear
markets by fading extremes during high-volume exhaustion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RSI_Reversal_with_Volume_and_Trend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: 24-period average (~12 days on 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(vol_ma_24[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 24-period average
        volume_filter = volume[i] > 2.0 * vol_ma_24[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        
        # Trend filter: use 1-day EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: fade extremes with volume and trend alignment
        long_entry = rsi_oversold and volume_filter and downtrend  # Buy weakness in downtrend
        short_entry = rsi_overbought and volume_filter and uptrend   # Sell strength in uptrend
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        long_exit = rsi[i] > 40
        short_exit = rsi[i] < 60
        
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
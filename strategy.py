#!/usr/bin/env python3
"""
12h_Range_Reversion_Triple_Filter
Strategy: Mean reversion at extreme RSI with volume confirmation and trend filter.
Long: RSI(14) < 20 + volume > 2x average + daily EMA34 > EMA144 (bullish)
Short: RSI(14) > 80 + volume > 2x average + daily EMA34 < EMA144 (bearish)
Exit: RSI crosses back to neutral (40-60) or trend reversal
Position size: 0.25
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    # Handle division by zero
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # Align with original index
    
    # Calculate 1d EMA34 and EMA144 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    close_series_1d = pd.Series(close_1d)
    ema34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema144_1d = close_series_1d.ewm(span=144, adjust=False, min_periods=144).mean().values
    
    # Align 1d EMAs to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema144_1d_aligned = align_htf_to_ltf(prices, df_1d, ema144_1d)
    
    # Volume confirmation (20-period MA on 12h)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(144, 20, 15)  # RSI needs 14+1, EMA needs 144
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema144_1d_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period average
        volume_filter = volume[i] > (2.0 * volume_ma20[i])
        
        # Trend filter: 1d EMA34 > EMA144 for long, < for short
        ema34_gt_ema144 = ema34_1d_aligned[i] > ema144_1d_aligned[i]
        ema34_lt_ema144 = ema34_1d_aligned[i] < ema144_1d_aligned[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 20
        rsi_overbought = rsi[i] > 80
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        if position == 0:
            # Long: oversold + uptrend + volume spike
            if rsi_oversold and ema34_gt_ema144 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: overbought + downtrend + volume spike
            elif rsi_overbought and ema34_lt_ema144 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or trend reverses
            if rsi_neutral or not ema34_gt_ema144:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or trend reverses
            if rsi_neutral or not ema34_lt_ema144:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Range_Reversion_Triple_Filter"
timeframe = "12h"
leverage = 1.0
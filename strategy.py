#!/usr/bin/env python3
"""
12h_1d_Range_Breakout_Volume_Trend
Hypothesis: Use daily range breakout with volume confirmation and 12h trend filter to capture breakout moves in both bull and bear markets. The strategy enters long when price breaks above the 1-day high with volume > 1.5x average and price above 12h EMA34 (uptrend), and short when price breaks below the 1-day low with volume > 1.5x average and price below 12h EMA34 (downtrend). Exits when price returns to the 12h EMA34 (trend exhaustion). Targets 15-25 trades/year by requiring alignment of daily breakout, volume confirmation, and trend filter. Works in bull markets by capturing upside breakouts and in bear markets by capturing downside breakdowns, with trend filter reducing whipsaws.
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
    
    # Get 1d data for daily high/low
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align daily high/low to 12h timeframe (wait for bar close)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Get 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema_12h[33] = np.mean(close_12h[:34])  # seed
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align EMA34 to 12h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # need volume MA and EMA seeded
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1-day high, with volume, and 12h uptrend (price > EMA34)
            if (close[i] > high_1d_aligned[i] and vol_confirm[i] and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1-day low, with volume, and 12h downtrend (price < EMA34)
            elif (close[i] < low_1d_aligned[i] and vol_confirm[i] and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below EMA34 (trend exhaustion)
            if close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above EMA34 (trend exhaustion)
            if close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Range_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
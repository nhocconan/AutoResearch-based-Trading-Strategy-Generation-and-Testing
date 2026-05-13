#!/usr/bin/env python3
"""
1h_4H_1D_Breakout_Volume_Trend
Hypothesis: Breakouts above 4h resistance or below 4h support with volume confirmation and 1d trend alignment capture momentum moves in both bull and bear markets. The 1h timeframe provides precise entry timing while 4h/1d filters reduce false signals and control trade frequency to avoid fee drag. Designed for 15-35 trades/year.
"""

name = "1h_4H_1D_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for support/resistance levels
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h resistance (20-period high) and support (20-period low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h resistance and support levels
    resistance_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    support_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h levels to 1h chart
    resistance_4h_aligned = align_htf_to_ltf(prices, df_4h, resistance_4h)
    support_4h_aligned = align_htf_to_ltf(prices, df_4h, support_4h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 24-period average (6 hours on 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above 4h resistance with volume confirmation and uptrend
            if (close[i] > resistance_4h_aligned[i] and 
                volume_filter[i] and 
                session_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below 4h support with volume confirmation and downtrend
            elif (close[i] < support_4h_aligned[i] and 
                  volume_filter[i] and 
                  session_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to 4h support or trend reverses
            if (close[i] < support_4h_aligned[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to 4h resistance or trend reverses
            if (close[i] > resistance_4h_aligned[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
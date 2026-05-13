#!/usr/bin/env python3
"""
12h_Pivot_Breakout_1dTrend_Volume
Hypothesis: 12h breakouts of 1-day high/low with 1d trend alignment and volume confirmation
capture momentum moves while avoiding false breakouts. Uses 1d high/low as support/resistance.
Exit on return to 1d close or trend reversal. Position size 0.25 targets ~20-30 trades/year.
"""

name = "12h_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot levels: 1d high and low as support/resistance
    pivot_high = high_1d
    pivot_low = low_1d
    
    # Trend filter: 20-period EMA on 1d close
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 12h timeframe
    pivot_high_aligned = align_htf_to_ltf(prices, df_1d, pivot_high)
    pivot_low_aligned = align_htf_to_ltf(prices, df_1d, pivot_low)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 12-period average (6 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Break above 1d high with volume and uptrend
            if (close[i] > pivot_high_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 1d low with volume and downtrend
            elif (close[i] < pivot_low_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Return to 1d close or trend reversal
            if (close[i] < close_1d[i//2] if i//2 < len(close_1d) else close_1d[-1]) or \
               (close[i] < ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Return to 1d close or trend reversal
            if (close[i] > close_1d[i//2] if i//2 < len(close_1d) else close_1d[-1]) or \
               (close[i] > ema20_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
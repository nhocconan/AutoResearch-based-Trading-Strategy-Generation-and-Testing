#!/usr/bin/env python3
"""
6h_Keltner_Channel_Breakout_With_Volume_Spike_v2
Hypothesis: Keltner Channel breakouts with volume spikes on 6h timeframe capture
momentum moves in trending markets. Uses 1w trend filter (EMA50) to ensure
alignment with higher timeframe trend, reducing false breakouts. Weekly trend
filter helps in both bull and bear markets by only taking trades in the
direction of the weekly trend. Position size 0.25 targets ~20-40 trades/year.
"""

name = "6h_Keltner_Channel_Breakout_With_Volume_Spike_v2"
timeframe = "6h"
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
    
    # Get 6h data for Keltner Channel calculation
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate Keltner Channel: EMA(20) ± ATR(10) * 2
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # EMA(20)
    ema20 = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(10)
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    kc_middle = ema20
    kc_upper = ema20 + (atr10 * 2)
    kc_lower = ema20 - (atr10 * 2)
    
    # Align Keltner Channel to 6h chart
    kc_middle_aligned = align_htf_to_ltf(prices, df_6h, kc_middle)
    kc_upper_aligned = align_htf_to_ltf(prices, df_6h, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_6h, kc_lower)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above upper KC with volume and weekly uptrend
            if (close[i] > kc_upper_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below lower KC with volume and weekly downtrend
            elif (close[i] < kc_lower_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle line or weekly trend reverses
            if (close[i] < kc_middle_aligned[i]) or \
               (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle line or weekly trend reverses
            if (close[i] > kc_middle_aligned[i]) or \
               (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
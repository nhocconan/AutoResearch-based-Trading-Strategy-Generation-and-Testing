#!/usr/bin/env python3
"""
1d_1w_Keltner_Channel_Breakout_With_Volume_Spike
Hypothesis: Keltner channel breakouts on daily timeframe capture strong trends in BTC/ETH.
Weekly EMA200 filters for trend alignment, and volume spikes confirm breakout strength.
Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull and bear markets.
"""

name = "1d_1w_Keltner_Channel_Breakout_With_Volume_Spike"
timeframe = "1d"
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
    
    # Get daily data for Keltner channel calculation
    df_1d = get_htf_data(prices, '1d')
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Keltner Channel: EMA(20) +/- 2 * ATR(10)
    ema20 = pd.Series(c_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr10 = pd.Series(h_1d - l_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_keltner = ema20 + 2.0 * atr10
    lower_keltner = ema20 - 2.0 * atr10
    
    # Align Keltner channels to daily chart
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume confirmation: current volume > 2.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Breakout above upper Keltner with volume confirmation and uptrend
            if (close[i] > upper_keltner_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below lower Keltner with volume confirmation and downtrend
            elif (close[i] < lower_keltner_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to EMA20 or trend reverses
            if (close[i] < ema20_aligned[i]) or \
               (close[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to EMA20 or trend reverses
            if (close[i] > ema20_aligned[i]) or \
               (close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
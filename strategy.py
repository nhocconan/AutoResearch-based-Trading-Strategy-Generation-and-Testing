#!/usr/bin/env python3
"""
6h_VolumeSpike_Reversal_1wTrend
Hypothesis: In the direction of the weekly trend (EMA200), enter on volume spikes (>2x 20-period average)
after a price pullback to the 50-period EMA. Exit when price crosses the 50 EMA in the opposite direction.
Designed for low trade frequency (~10-20/year) to minimize fee drag. Volume spikes indicate
institutional participation, and buying pullbacks in an uptrend (or selling rallies in a downtrend)
is a proven edge. Weekly trend filter ensures we only trade with the dominant multi-week momentum.
"""

name = "6h_VolumeSpike_Reversal_1wTrend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 50 EMA for pullback entries (6h timeframe)
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter: >2x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(ema50[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (price > weekly EMA200), price at 50 EMA (pullback), volume spike
            if (close[i] > ema200_1w_aligned[i] and  # Weekly uptrend
                abs(close[i] - ema50[i]) / ema50[i] < 0.005 and  # Within 0.5% of 50 EMA (pullback)
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < weekly EMA200), price at 50 EMA (pullback), volume spike
            elif (close[i] < ema200_1w_aligned[i] and  # Weekly downtrend
                  abs(close[i] - ema50[i]) / ema50[i] < 0.005 and  # Within 0.5% of 50 EMA (pullback)
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 50 EMA (trend invalidation)
            if close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 50 EMA (trend invalidation)
            if close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
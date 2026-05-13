#!/usr/bin/env python3
"""
6h_Elder_Ray_Power_Divergence_1dTrend_Filter
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
Combined with 1d trend filter (close > EMA50) and volume confirmation, it captures strong directional moves
while avoiding weak/choppy markets. Works in both bull and bear trends by following the 1d trend.
Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

name = "6h_Elder_Ray_Power_Divergence_1dTrend_Filter"
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
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values
    bear_power = ema13.values - low
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after EMA13 warmup
        if position == 0:
            # LONG: Bull Power rising (positive divergence), price above 1d EMA50, volume confirmation
            if (bull_power[i] > bull_power[i-1] and  # Rising bull power
                bull_power[i] > 0 and                 # Actually bullish
                close[i] > ema50_1d_aligned[i] and    # Uptrend filter
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power rising (positive divergence), price below 1d EMA50, volume confirmation
            elif (bear_power[i] > bear_power[i-1] and   # Rising bear power
                  bear_power[i] > 0 and                 # Actually bearish
                  close[i] < ema50_1d_aligned[i] and    # Downtrend filter
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power falling OR price crosses below 1d EMA50
            if (bull_power[i] < bull_power[i-1]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power falling OR price crosses above 1d EMA50
            if (bear_power[i] < bear_power[i-1]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
6h_Volume_Imbalance_Reversal
Hypothesis: Extreme volume spikes on 6h charts indicate exhaustion points. 
Buy when volume spikes 3x average AND price is below VWAP (oversold), 
sell when volume spikes 3x average AND price is above VWAP (overbought).
Uses 1d trend filter to avoid counter-trend trades. Targets 15-25 trades/year.
Works in both bull/bear markets by fading exhaustion moves.
"""

name = "6h_Volume_Imbalance_Reversal"
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
    
    # Calculate VWAP for each 6h bar
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Volume spike detection: current volume > 3.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * vol_ma)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Volume spike + price below VWAP (oversold) + uptrend
            if (volume_spike[i] and 
                close[i] < vwap[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Volume spike + price above VWAP (overbought) + downtrend
            elif (volume_spike[i] and 
                  close[i] > vwap[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to VWAP or trend reverses
            if (close[i] >= vwap[i]) or \
               (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP or trend reverses
            if (close[i] <= vwap[i]) or \
               (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
4h_Supertrend_Direction_With_Volume_Spike
Hypothesis: Supertrend (ATR-based trend filter) provides reliable directional bias. 
When price breaks above/below Supertrend with volume confirmation and aligned 1d trend (close > EMA50), 
it signals continuation. Uses 25% position size to balance risk/return and limit trade frequency (~20-40/year) 
to minimize fee drag in 4-hour bars. Works in both bull and bear markets by following the trend.
"""

name = "4h_Supertrend_Direction_With_Volume_Spike"
timeframe = "4h"
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Supertrend (10, 3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Final Upper and Lower Bands
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    final_upper[0] = upper_band[0]
    final_lower[0] = lower_band[0]
    
    for i in range(1, n):
        if close[i-1] <= final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = min(upper_band[i], final_upper[i-1])
            
        if close[i-1] >= final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = max(lower_band[i], final_lower[i-1])
    
    # Supertrend
    supertrend = np.zeros(n)
    supertrend[0] = final_lower[0]
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > final_upper[i-1]:
            direction[i] = 1
        elif close[i] < final_lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
    
    # 1d trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        if position == 0:
            # LONG: Price above Supertrend (uptrend), volume confirmation, price above 1d EMA50
            if (close[i] > supertrend[i] and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Supertrend (downtrend), volume confirmation, price below 1d EMA50
            elif (close[i] < supertrend[i] and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below Supertrend (trend change) OR volume drops
            if (close[i] < supertrend[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above Supertrend (trend change) OR volume drops
            if (close[i] > supertrend[i]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
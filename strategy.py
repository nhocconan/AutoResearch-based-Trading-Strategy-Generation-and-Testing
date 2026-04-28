#!/usr/bin/env python3
"""
4H_4H_CCI_Trend_Filter_12H_Volume_Spike
Hypothesis: On 4H timeframe, enter long when CCI crosses above +100 with 12H uptrend and volume spike, short when CCI crosses below -100 with 12H downtrend and volume spike. Exit on CCI crossing back through zero. The 12H trend filter ensures alignment with higher timeframe momentum, while volume spikes confirm institutional participation. CCI captures overbought/oversold conditions with mean reversion tendencies. Designed for moderate trade frequency (~20-40/year) to balance signal quality and fee efficiency in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12H data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12H EMA21 for trend filter
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 12H EMA21 to 4H timeframe
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Calculate CCI(20) on 4H data
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for CCI calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: CCI crosses extreme levels with trend alignment and volume spike
        long_entry = (cci[i] > 100 and cci[i-1] <= 100 and ema21_12h_aligned[i] > close[i] and volume_spike[i])
        short_entry = (cci[i] < -100 and cci[i-1] >= -100 and ema21_12h_aligned[i] < close[i] and volume_spike[i])
        
        # Exit conditions: CCI crosses back through zero
        long_exit = (cci[i] < 0 and cci[i-1] >= 0)
        short_exit = (cci[i] > 0 and cci[i-1] <= 0)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Close long
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.25   # Close short
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4H_4H_CCI_Trend_Filter_12H_Volume_Spike"
timeframe = "4h"
leverage = 1.0
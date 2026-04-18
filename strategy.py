# 12h_Price_Action_Breakout_V1
# Hypothesis: On 12h timeframe, price breaking out of daily high/low with volume confirmation
# captures institutional moves in both bull and bear markets. Uses daily high/low as natural
# support/resistance levels. Volume filter reduces false breakouts. Designed for low trade
# frequency to avoid fee drag.

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
    
    # Get daily data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Align daily levels to 12h timeframe (use previous day's levels)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 30:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-30+1:i+1])
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above daily high with volume spike
            if close[i] > daily_high_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily low with volume spike
            elif close[i] < daily_low_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below daily low (mean reversion) or volume dies
            if close[i] < daily_low_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above daily high or volume dies
            if close[i] > daily_high_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Price_Action_Breakout_V1"
timeframe = "12h"
leverage = 1.0
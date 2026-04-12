# 12h_1d_volume_breakout
# Hypothesis: Use daily (1d) volume-weighted price action as trend filter. On 12h timeframe,
# enter long when price breaks above daily VWAP with volume expansion, short when breaks below.
# VWAP acts as dynamic support/resistance. Volume confirmation ensures institutional interest.
# Works in bull (breakouts continue) and bear (breakdowns continue) markets.
# Target: 15-25 trades/year to minimize fee drag.

name = "12h_1d_volume_breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP: typical price * volume / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_raw = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    
    # Align daily VWAP to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_raw.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if VWAP not ready
        if np.isnan(vwap_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: price breaks above VWAP with volume expansion
        if (close[i] > vwap_aligned[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price breaks below VWAP with volume expansion
        elif (close[i] < vwap_aligned[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price crosses back through VWAP
        elif position == 1 and close[i] < vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > vwap_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
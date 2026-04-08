#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_breakout_volume_v1
Hypothesis: Use 1d Camarilla pivot levels for support/resistance and 1w trend direction. 
Enter long when price breaks above H3 with volume confirmation in uptrend, short when breaks below L3 in downtrend.
Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) markets.
Target: 12-37 trades/year per symbol (48-148 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    H3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    L3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1w EMA(21) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x average of last 24 periods
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 or trend turns bearish
            if close[i] < L3_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price breaks above H3 or trend turns bullish
            if close[i] > H3_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above H3 with volume confirmation in uptrend
            if (close[i] > H3_aligned[i] and 
                close[i] > ema_1w_aligned[i] and  # Uptrend filter
                vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume confirmation in downtrend
            elif (close[i] < L3_aligned[i] and 
                  close[i] < ema_1w_aligned[i] and  # Downtrend filter
                  vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
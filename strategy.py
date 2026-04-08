# 4h_12h_camarilla_pivot_volume_v2
# Hypothesis: Camarilla pivot levels from 12h + volume confirmation + 4h trend filter (EMA21) captures institutional reversal points in both bull and bear markets. 4h timeframe limits overtrading; pivot levels provide structured support/resistance; volume confirms institutional participation.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_pivot_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formulas
    range_12h = high_12h - low_12h
    camarilla_h4 = close_12h + range_12h * 1.1 / 2
    camarilla_l4 = close_12h - range_12h * 1.1 / 2
    camarilla_h3 = close_12h + range_12h * 1.1 / 4
    camarilla_l3 = close_12h - range_12h * 1.1 / 4
    
    # Align to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
    # 4h EMA21 for trend filter
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema21[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Camarilla L3
            if close[i] < l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above Camarilla H3
            if close[i] > h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price touches Camarilla L4, above EMA21, with volume confirmation
            if (low[i] <= l4_aligned[i] and close[i] > ema21[i] and vol_confirm[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches Camarilla H4, below EMA21, with volume confirmation
            elif (high[i] >= h4_aligned[i] and close[i] < ema21[i] and vol_confirm[i]):
                position = -1
                signals[i] = -0.25
    
    return signals
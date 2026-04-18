#!/usr/bin/env python3
"""
4h_HTF_Trend_With_Volume_Spike_Entry_v1
Hypothesis: Use 12h EMA34 as primary trend filter with 4h price action for entry timing. Enter long when 4h close crosses above 12h EMA34 with volume spike (>2x 20-period average), short when crossing below. Exit when price crosses back over 12h EMA34. This captures medium-term trends while minimizing whipsaw through volume confirmation. Designed for 20-40 trades/year to avoid fee drag.
"""

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
    
    # 12h EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(volume_spike[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_price = close[i-1]
        ema_val = ema_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price crosses above EMA with volume spike
            if prev_price <= ema_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below EMA with volume spike
            elif prev_price >= ema_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below EMA
            if price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above EMA
            if price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_HTF_Trend_With_Volume_Spike_Entry_v1"
timeframe = "4h"
leverage = 1.0
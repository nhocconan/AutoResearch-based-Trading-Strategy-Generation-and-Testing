#!/usr/bin/env python3
"""
1h_price_action_volume_confluence_v1
Hypothesis: On 1h timeframe, price action (close relative to open) combined with 
volume confirmation and 4h trend filter (EMA50) creates high-probability entries.
In both bull and bear markets, we look for strong closes with volume in the 
direction of the 4h trend. Time-based filter (08-20 UTC) reduces noise.
Target: 15-37 trades/year per symbol by requiring volume > 2x average and 
strong price action (>60% of range). Max 200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_price_action_volume_confluence_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute hour filter
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Price action: close position in range
        rng = high[i] - low[i]
        if rng == 0:
            close_pos = 0.5
        else:
            close_pos = (close[i] - low[i]) / rng
        
        # Volume confirmation: > 2x average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Strong close: >60% of range in direction
        strong_close_up = close_pos > 0.6
        strong_close_down = close_pos < 0.4
        
        if position == 1:  # Long position
            # Exit: weak close or volume drops
            if not (strong_close_up and vol_confirm):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: weak close or volume drops
            if not (strong_close_down and vol_confirm):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: strong up close + volume + 4h uptrend
            if (strong_close_up and 
                vol_confirm and 
                close[i] > ema50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short: strong down close + volume + 4h downtrend
            elif (strong_close_down and 
                  vol_confirm and 
                  close[i] < ema50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals
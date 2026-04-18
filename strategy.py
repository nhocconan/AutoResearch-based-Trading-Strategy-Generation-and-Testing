#!/usr/bin/env python3
"""
12h EMA Crossover + Volume Spike + 1w Trend Filter
Hypothesis: In both bull and bear markets, EMA crossovers on 12h timeframe combined with volume spikes and 1-week trend filtering provide reliable signals. 
The 1-week EMA acts as a strong trend filter to avoid counter-trend trades, while volume spikes confirm institutional participation.
Designed for low frequency (12-30 trades/year) with clear entry/exit rules.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter
    weekly_close = df_w['close'].values
    ema_20_w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to 12h timeframe
    ema_20_w_aligned = align_htf_to_ltf(prices, df_w, ema_20_w)
    
    # Calculate 12h EMA9 and EMA21 for crossover
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for EMA calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_w_aligned[i]) or 
            np.isnan(ema_9[i]) or
            np.isnan(ema_21[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_9_val = ema_9[i]
        ema_21_val = ema_21[i]
        ema_20_w_val = ema_20_w_aligned[i]
        
        if position == 0:
            # Long: EMA9 crosses above EMA21 with volume spike and above weekly EMA20
            if (ema_9_val > ema_21_val and ema_9[i-1] <= ema_21[i-1] and 
                volume_spike[i] and price > ema_20_w_val):
                signals[i] = 0.25
                position = 1
            # Short: EMA9 crosses below EMA21 with volume spike and below weekly EMA20
            elif (ema_9_val < ema_21_val and ema_9[i-1] >= ema_21[i-1] and 
                  volume_spike[i] and price < ema_20_w_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: EMA9 crosses below EMA21 or price drops below weekly EMA20
            if (ema_9_val < ema_21_val and ema_9[i-1] >= ema_21[i-1]) or price < ema_20_w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: EMA9 crosses above EMA21 or price rises above weekly EMA20
            if (ema_9_val > ema_21_val and ema_9[i-1] <= ema_21[i-1]) or price > ema_20_w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_EMACrossover_VolumeSpike_1wEMA20"
timeframe = "12h"
leverage = 1.0
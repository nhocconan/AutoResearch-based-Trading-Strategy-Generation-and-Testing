#!/usr/bin/env python3
"""
12h_TRIX_Volume_Spike_1dTrend_v1
Hypothesis: On 12h timeframe, use TRIX (12-period) to identify momentum direction and strength,
filtered by daily EMA trend and volume spikes to avoid false signals.
Long when TRIX > 0 and price above daily EMA with volume spike.
Short when TRIX < 0 and price below daily EMA with volume spike.
TRIX is a momentum oscillator that filters out insignificant price movements,
making it effective in both trending and ranging markets when combined with trend and volume filters.
"""
name = "12h_TRIX_Volume_Spike_1dTrend_v1"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate TRIX (12-period)
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1 period ago
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change(1) * 100  # Percentage change
    trix_values = trix.values
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(36, 20)  # TRIX needs 36 bars (12*3), volume needs 20
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data is not ready
        if (np.isnan(trix_values[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        if position == 0:
            # Minimum 4 bars between trades to reduce frequency (12h timeframe)
            if bars_since_entry < 4:
                continue
                
            # Long: TRIX > 0 (bullish momentum) + price above EMA34 + volume filter
            if (trix_values[i] > 0 and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: TRIX < 0 (bearish momentum) + price below EMA34 + volume filter
            elif (trix_values[i] < 0 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position != 0:
            # Exit: TRIX crosses zero in opposite direction
            if position == 1:
                if trix_values[i] < 0:  # Bearish crossover
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if trix_values[i] > 0:  # Bullish crossover
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals
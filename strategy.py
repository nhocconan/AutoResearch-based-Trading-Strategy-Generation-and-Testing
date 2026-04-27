#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
# Long when price breaks above 1h high of prior 4h bar with 4h uptrend and volume spike (>1.5x avg).
# Short when price breaks below 1h low of prior 4h bar with 4h downtrend and volume spike.
# Uses 4h structure to limit trades to ~20-40 per year, targeting 80-160 total over 4 years.
# Volume spike filters for institutional participation, reducing false breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for structure and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h high/low for structure (prior 4h bar)
    structure_high = high_4h  # High of completed 4h bar
    structure_low = low_4h    # Low of completed 4h bar
    
    # Align 4h structure to 1h timeframe
    structure_high_aligned = align_htf_to_ltf(prices, df_4h, structure_high)
    structure_low_aligned = align_htf_to_ltf(prices, df_4h, structure_low)
    
    # 20-period EMA on 4h close for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(structure_high_aligned[i]) or np.isnan(structure_low_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above 4h high AND 4h uptrend AND volume spike
        if (close[i] > structure_high_aligned[i] and 
            close[i] > ema20_4h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.20
            position = 1
        # Short conditions: price breaks below 4h low AND 4h downtrend AND volume spike
        elif (close[i] < structure_low_aligned[i] and 
              close[i] < ema20_4h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.20
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4hStructure_Breakout_20EMA_VolumeFilter"
timeframe = "1h"
leverage = 1.0
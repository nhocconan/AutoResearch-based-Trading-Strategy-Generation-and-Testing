# 1d_1d_WoO_Range_Breakout_Volume_Confirm
# Hypothesis: Trade breakouts of the previous day's range (high/low) with volume confirmation on the daily chart.
# Works in both bull and bear markets by capturing momentum after range breaks. Uses 1d timeframe for structure and volume.
# Volume confirmation filters out false breakouts. Position size 0.25 to manage drawdown.
# Expects ~10-25 trades/year, well within limits to avoid fee drag.

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
    
    # Get 1d data for previous day's range and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's high/low (shifted by 1 to avoid look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Previous day's volume (for comparison)
    prev_volume_1d = np.roll(volume_1d, 1)
    prev_volume_1d[0] = np.nan
    
    # Align to 1d timeframe (no alignment needed as we're already on 1d)
    # But we use the values directly since timeframe is 1d
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # 25% position size
    
    # Start from index 1 (need previous day)
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_1d[i]) or np.isnan(prev_low_1d[i]) or
            np.isnan(prev_volume_1d[i])):
            continue
        
        # Long: break above prev day high with volume > 1.5x prev day volume
        if (high[i] > prev_high_1d[i] and
            volume[i] > 1.5 * prev_volume_1d[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: break below prev day low with volume > 1.5x prev day volume
        elif (low[i] < prev_low_1d[i] and
              volume[i] > 1.5 * prev_volume_1d[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse break of the opposite level
        elif position == 1 and low[i] < prev_low_1d[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > prev_high_1d[i]:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_1d_WoO_Range_Breakout_Volume_Confirm"
timeframe = "1d"
leverage = 1.0
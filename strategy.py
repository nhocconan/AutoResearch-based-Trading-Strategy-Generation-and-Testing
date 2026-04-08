#!/usr/bin/env python3
# 4h_12h_volume_surge_breakout_v1
# Hypothesis: 4-hour price breaks Donchian channel (20) with 12-hour volume surge (>2x average) confirms institutional interest.
# Long: price breaks above Donchian(20) high with volume surge. Short: breaks below low with volume surge.
# Exit: price returns to Donchian midpoint or opposite breakout.
# Volume surge filters false breakouts; Donchian provides objective trend framework.
# Designed for 20-40 trades/year on 4h to minimize fee drag while capturing significant moves.
# Works in bull markets via upward breaks and bear markets via downward breaks.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_volume_surge_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Get 12h volume data for confirmation
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # 12h volume moving average (20-period) for surge detection
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    vol_current_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure Donchian is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_current_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current 12h volume > 2x 20-period average
        vol_surge = vol_current_aligned[i] > 2.0 * vol_ma_20_aligned[i] and vol_ma_20_aligned[i] > 0
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or breaks below Donchian low
            if close[i] <= donch_mid[i] or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or breaks above Donchian high
            if close[i] >= donch_mid[i] or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume surge
            if close[i] > donch_high[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume surge
            elif close[i] < donch_low[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals
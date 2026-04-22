#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h price action strategy using 1d high/low channels with volume confirmation
    # Works in both bull and bear markets: buys near 1d support, sells near 1d resistance
    # Uses 1-day high/low channels as dynamic support/resistance levels
    # Volume surge confirms institutional interest at key levels
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day channel: 20-period high/low (dynamic support/resistance)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1-day channel to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20
    
    # Price position within 1-day channel (0 = at low, 1 = at high)
    price_pos = (prices['close'].values - low_20_aligned) / (high_20_aligned - low_20_aligned + 1e-10)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Near 1d support (low channel) with volume surge
            if price_pos[i] < 0.2 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Near 1d resistance (high channel) with volume surge
            elif price_pos[i] > 0.8 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price moves to middle of channel or opposite extreme
            if position == 1:
                if price_pos[i] > 0.5:  # Moved to middle
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_pos[i] < 0.5:  # Moved to middle
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_1dChannel_Volume_Surge_v1"
timeframe = "12h"
leverage = 1.0
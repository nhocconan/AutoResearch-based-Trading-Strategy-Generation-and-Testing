#!/usr/bin/env python3
"""
12h_1d_donchian_breakout_volume_regime
Hypothesis: 12-hour Donchian breakout with volume confirmation and 1-day chop regime filter.
Works in bull/bear by filtering breakouts with chop regime and volume confirmation.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

name = "12h_1d_donchian_breakout_volume_regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian channels (20-period)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ATR for chop calculation (14-period)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation: sum of true range over 14 periods / (max(high) - min(low) over 14 periods)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = 100 * np.divide(sum_tr_14, range_14, out=np.full_like(sum_tr_14, 50.0), where=range_14!=0)
    
    # Align indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volume confirmation: volume > 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: chop > 61.8 = ranging (mean revert), chop < 38.2 = trending
        # We'll use chop < 61.8 to allow both trending and mild ranging
        chop_filter = chop_aligned[i] < 61.8
        
        # Long entry: close breaks above Donchian high with volume and chop filter
        if (close[i] > donch_high_aligned[i] and vol_confirm[i] and 
            chop_filter and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below Donchian low with volume and chop filter
        elif (close[i] < donch_low_aligned[i] and vol_confirm[i] and 
              chop_filter and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite Donchian level
        elif position == 1 and close[i] < donch_low_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > donch_high_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
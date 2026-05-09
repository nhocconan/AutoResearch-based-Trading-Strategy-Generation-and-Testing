#!/usr/bin/env python3
# Hypothesis: 12h Choppiness Index regime filter combined with 1d Donchian breakout and volume confirmation
# In trending markets (CHOP < 38.2): trade breakouts of 1d Donchian channels
# In ranging markets (CHOP > 61.8): fade moves at Donchian extremes with mean reversion
# Volume confirmation filters low-quality breakouts
# Designed to work in both bull and bear markets by adapting to market regime
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "12h_ChoppinessRegime_DonchianBreakout_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 1d Choppiness Index for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR(14)
    atr14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Sum of true ranges over 14 periods
    tr_sum = tr.rolling(window=14, min_periods=14).sum()
    
    # Choppiness Index: 100 * log10(tr_sum / (atr14 * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr14 * 14)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 1d Donchian channels (20-period)
    donch_high = df_1d['high'].rolling(window=20, min_periods=20).max().values
    donch_low = df_1d['low'].rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.3 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        
        if position == 0:
            # Trending market: CHOP < 38.2 - trade breakouts
            if chop_val < 38.2:
                # Long breakout above Donchian high with volume
                if close[i] > donch_high_val and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below Donchian low with volume
                elif close[i] < donch_low_val and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8 - fade extremes
            elif chop_val > 61.8:
                # Short at Donchian high (expect reversion to mean)
                if close[i] > donch_high_val:
                    signals[i] = -0.25
                    position = -1
                # Long at Donchian low (expect reversion to mean)
                elif close[i] < donch_low_val:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Exit long: chop becomes too high (range) or price hits opposite Donchian band
            if chop_val > 61.8 or close[i] < donch_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: chop becomes too high (range) or price hits opposite Donchian band
            if chop_val > 61.8 or close[i] > donch_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
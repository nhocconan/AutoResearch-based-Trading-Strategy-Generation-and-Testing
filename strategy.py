#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout + volume confirmation
# Uses Choppiness Index (14-period) to identify trending vs ranging markets
# In trending markets (CHOP < 38.2): trade Donchian breakouts
# In ranging markets (CHOP > 61.8): mean-revert at Donchian channels
# Volume confirmation (>1.5x average) reduces false signals
# 1d Donchian provides higher timeframe structure
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_Chop_Donchian_1d_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Choppiness Index (14-period) on 4h data
    def true_range(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First period
        return tr
    
    tr = true_range(high, low, close)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop = np.where((hh_14 - ll_14) != 0, chop_raw, 50.0)  # Default to neutral when range=0
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market: CHOP < 38.2
            if chop[i] < 38.2:
                # Long breakout: price breaks above 1d Donchian high
                if close[i] > donch_high_aligned[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price breaks below 1d Donchian low
                elif close[i] < donch_low_aligned[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: CHOP > 61.8
            elif chop[i] > 61.8:
                # Mean reversion long: price near Donchian low
                if close[i] <= donch_low_aligned[i] * 1.005 and volume_confirm[i]:  # Within 0.5% of low
                    signals[i] = 0.25
                    position = 1
                # Mean reversion short: price near Donchian high
                elif close[i] >= donch_high_aligned[i] * 0.995 and volume_confirm[i]:  # Within 0.5% of high
                    signals[i] = -0.25
                    position = -1
                
        elif position == 1:
            # Long exit: price reaches Donchian high or choppiness exits trending state
            if (close[i] >= donch_high_aligned[i] * 0.995) or (chop[i] > 50.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price reaches Donchian low or choppiness exits trending state
            if (close[i] <= donch_low_aligned[i] * 1.005) or (chop[i] > 50.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
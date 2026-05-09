#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation
# Uses Choppiness Index to filter ranging markets (avoid whipsaw) and breakouts on higher timeframe
# to capture trending moves. Works in both bull/bear by following 1d Donchian breakouts.
# Choppiness Index > 61.8 = ranging (mean-reversion), < 38.2 = trending (trend-following).
# We only take breakout signals when market is trending (CHOP < 38.2) with volume confirmation.
name = "4h_ChopFilter_1dDonchian_Breakout_Volume"
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
    
    # 1d data for Donchian channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Choppiness Index (14-period) on 4h data
    def true_range(high, low, close):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First TR is just high-low
        return tr
    
    tr = true_range(high, low, close)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_high_low = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low / (atr14 * 14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((atr14 * 14) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # 1d Donchian channels (20-period)
    donch_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(chop[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Only trade when market is trending (CHOP < 38.2)
            if chop[i] < 38.2:
                # Long: price breaks above 1d Donchian high + volume
                if price > donch_high_aligned[i] and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below 1d Donchian low + volume
                elif price < donch_low_aligned[i] and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price breaks below 1d Donchian low or chop increases (ranging market)
            if price < donch_low_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 1d Donchian high or chop increases (ranging market)
            if price > donch_high_aligned[i] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
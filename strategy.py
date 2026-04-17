#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above 4h Donchian upper band with volume confirmation and price > 1d EMA50 (uptrend).
Short when price breaks below 4h Donchian lower band with volume confirmation and price < 1d EMA50 (downtrend).
Exit when price returns to the 4h Donchian midpoint or reverses with volume.
Uses 4h/1d for signal direction (reduces noise) and 1h only for entry timing precision.
Designed to capture medium-term breakouts with institutional volume while avoiding false breakouts in choppy markets.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1h volume 20-period average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with volume and uptrend (price > EMA50)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with volume and downtrend (price < EMA50)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below lower with volume (reversal)
            if (close[i] <= donchian_mid_aligned[i] or 
                (close[i] < donchian_lower_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above upper with volume (reversal)
            if (close[i] >= donchian_mid_aligned[i] or 
                (close[i] > donchian_upper_aligned[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hDonchian20_Breakout_Volume_1dEMA50_Trend"
timeframe = "1h"
leverage = 1.0
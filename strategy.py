#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above 1d Donchian upper channel (20-day high) with 1d volume > 1.5x 20-day average and price > 1w EMA34.
Short when price breaks below 1d Donchian lower channel (20-day low) with 1d volume > 1.5x 20-day average and price < 1w EMA34.
Exit when price returns to the 1d Donchian midpoint or reverses with volume confirmation.
Uses 1d for price channels and volume regime, 1w for trend filter.
Designed to capture strong trending moves with volume confirmation in both bull and bear markets.
Target: 15-25 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    dc_upper = high_20
    dc_lower = low_20
    dc_mid = (dc_upper + dc_lower) / 2.0
    
    # Calculate 1d volume MA20 for regime filter
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # need enough for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or 
            np.isnan(dc_mid[i]) or 
            np.isnan(vol_ma_20_1d[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1d volume > 1.5x 20-day average (expanding participation)
        volume_confirmed = volume_1d[i] > 1.5 * vol_ma_20_1d[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume confirmation and uptrend (price > EMA34)
            if (close_1d[i] > dc_upper[i] and 
                volume_confirmed and 
                close_1d[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume confirmation and downtrend (price < EMA34)
            elif (close_1d[i] < dc_lower[i] and 
                  volume_confirmed and 
                  close_1d[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below midpoint OR breaks below lower with volume (reversal)
            if (close_1d[i] <= dc_mid[i] or 
                (close_1d[i] < dc_lower[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above midpoint OR breaks above upper with volume (reversal)
            if (close_1d[i] >= dc_mid[i] or 
                (close_1d[i] > dc_upper[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume_Regime"
timeframe = "1d"
leverage = 1.0
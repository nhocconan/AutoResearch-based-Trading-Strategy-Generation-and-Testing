#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume regime and 1d EMA50 trend filter.
Long when price breaks above 20-period Donchian high with 1d volume > 1.5x 20-day average and price > 1d EMA50.
Short when price breaks below 20-period Donchian low with 1d volume > 1.5x 20-day average and price < 1d EMA50.
Exit when price returns to the midpoint of the Donchian channel or reverses with volume confirmation.
Uses 1d for trend and volume regime, 12h for execution.
Designed to capture medium-term breakouts with volume confirmation in both bull and bear markets.
Volume regime filter ensures trades only occur during periods of higher participation, reducing whipsaws.
Target: 12-37 trades/year per symbol to minimize fee drag.
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
    
    # Get 1d data for EMA50 trend filter and volume regime
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume MA20 for regime filter
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 12h volume > 1.5x 20-period MA (expanding participation)
        vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20_12h[i]) and volume[i] > 1.5 * vol_ma_20_12h[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and uptrend (price > EMA50)
            if (close[i] > donchian_high[i] and 
                volume_confirmed and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume confirmation and downtrend (price < EMA50)
            elif (close[i] < donchian_low[i] and 
                  volume_confirmed and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR breaks below Donchian low with volume (reversal)
            if (close[i] <= donchian_mid[i] or 
                (close[i] < donchian_low[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR breaks above Donchian high with volume (reversal)
            if (close[i] >= donchian_mid[i] or 
                (close[i] > donchian_high[i] and volume_confirmed)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeRegime"
timeframe = "12h"
leverage = 1.0
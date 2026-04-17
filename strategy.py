#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h volume regime filter and 1d ADX trend confirmation.
Long when price breaks above 20-period Donchian high AND 12h volume ratio > 1.5 (high volume regime) AND 1d ADX > 25 (trending market).
Short when price breaks below 20-period Donchian low AND 12h volume ratio > 1.5 AND 1d ADX > 25.
Exit when price touches the 20-period Donchian midpoint or opposite band.
Uses 12h for volume regime, 1d for ADX trend, 6h for execution and Donchian calculation.
Designed to capture strong breakouts in high-volume trending markets. Target: 15-25 trades/year per symbol.
"""

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
    
    # Get 12h data for volume regime
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for ADX trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h volume MA for regime filter
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / np.where(vol_ma_20_12h == 0, 1, vol_ma_20_12h)  # avoid div by zero
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(abs(high_1d[1:] - high_1d[:-1]), 
                               abs(low_1d[1:] - low_1d[:-1])))
    # Pad first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_14 == 0, 1, atr_14)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_14 == 0, 1, atr_14)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) == 0, 1, (plus_di_14 + minus_di_14))
    adx_14 = pd.Series(dx_14).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or
            np.isnan(donch_mid[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: 12h volume ratio > 1.5 (high volume)
        volume_regime = vol_ratio_12h_aligned[i] > 1.5
        
        # Trend filter: 1d ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > donch_high[i]
        breakout_down = close[i] < donch_low[i]
        
        # Exit conditions: touch midpoint or opposite band
        touch_mid = abs(close[i] - donch_mid[i]) < 0.001 * close[i]  # within 0.1%
        touch_opposite = (position == 1 and close[i] < donch_low[i]) or \
                         (position == -1 and close[i] > donch_high[i])
        
        if position == 0:
            # Long: break above Donchian high with volume regime and trending market
            if (breakout_up and volume_regime and trending):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume regime and trending market
            elif (breakout_down and volume_regime and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch midpoint or break below Donchian low
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch midpoint or break above Donchian high
            if (touch_mid or touch_opposite):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_VolumeRegime_ADXTrend"
timeframe = "6h"
leverage = 1.0
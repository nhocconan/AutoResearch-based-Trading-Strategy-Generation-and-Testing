#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(40) breakout + volume spike (2x 20-period average) + ADX(14) > 25.
Long when price breaks above 1d Donchian high with volume confirmation and strong trend.
Short when price breaks below 1d Donchian low with volume confirmation and strong trend.
Exit when price reverses to opposite Donchian level or ADX weakens (<20).
Using longer Donchian period (40) for fewer, more significant breakouts. Volume spike filter reduces false signals.
Target: 50-150 total trades over 4 years (12-37/year) to stay within fee-efficient range.
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
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian Channel (40) - longer period for fewer, stronger signals
    period40_high = pd.Series(high_1d).rolling(window=40, min_periods=40).max().values
    period40_low = pd.Series(low_1d).rolling(window=40, min_periods=40).min().values
    donchian_high = period40_high
    donchian_low = period40_low
    
    # Calculate 1d ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume 20-period average for volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available (NaN)
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 20-period average (volume spike)
        volume_spike = volume_1d_aligned[i] > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume spike and strong trend
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume spike and strong trend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend weakens (ADX < 20)
            if (close[i] < donchian_low_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend weakens (ADX < 20)
            if (close[i] > donchian_high_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian40_VolumeSpike_ADX"
timeframe = "12h"
leverage = 1.0
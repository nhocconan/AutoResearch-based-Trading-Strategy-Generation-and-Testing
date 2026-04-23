#!/usr/bin/env python3
"""
12h Weekly Donchian Breakout with Volume Spike and ADX Trend Filter
Long when price breaks above weekly Donchian high (20 periods) with volume > 1.5x avg volume and ADX > 20
Short when price breaks below weekly Donchian low (20 periods) with volume > 1.5x avg volume and ADX > 20
Exit when price returns to weekly Donchian midpoint or ADX < 15
Designed for low frequency (~20-40 trades/year) to capture major trends while avoiding whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels and ADX - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian high and low
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate weekly ADX (14-period)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Align weekly indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low.values)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid.values)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx.values)
    
    # Calculate average volume for volume spike filter (20-period weekly avg volume aligned)
    avg_vol_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean()
    avg_vol_aligned = align_htf_to_ltf(prices, df_1w, avg_vol_1w.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(avg_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current volume > 1.5x weekly average volume
        volume_spike = volume[i] > 1.5 * avg_vol_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume spike and ADX > 20
            if (close[i] > donch_high_aligned[i] and volume_spike and adx_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume spike and ADX > 20
            elif (close[i] < donch_low_aligned[i] and volume_spike and adx_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to weekly Donchian midpoint or ADX < 15
                if (close[i] >= donch_mid_aligned[i] or adx_aligned[i] < 15):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to weekly Donchian midpoint or ADX < 15
                if (close[i] <= donch_mid_aligned[i] or adx_aligned[i] < 15):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WeeklyDonchian_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0
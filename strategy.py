#!/usr/bin/env python3
"""
Weekly Donchian Breakout with Daily Volume Confirmation and ADX Trend Filter.
Long when price breaks above weekly Donchian upper channel with daily volume spike and ADX > 25.
Short when price breaks below weekly Donchian lower channel with daily volume spike and ADX > 25.
Exit when price crosses back to the weekly Donchian middle line (average of upper/lower).
Designed for low trade frequency with strong trend confirmation.
Works in both bull and bear markets by following weekly trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period Donchian channels on weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high_20 + donch_low_20) / 2.0
    
    # Align weekly channels to daily timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    
    # Daily ADX for trend filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_14[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above weekly Donchian upper + volume spike + ADX > 25
            if (close[i] > donch_high_20_aligned[i] and vol_spike and adx_14[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian lower + volume spike + ADX > 25
            elif (close[i] < donch_low_20_aligned[i] and vol_spike and adx_14[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to weekly Donchian middle
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below weekly Donchian middle
                if close[i] < donch_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above weekly Donchian middle
                if close[i] > donch_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Weekly_Donchian_Breakout_DailyVolume_ADX"
timeframe = "1d"
leverage = 1.0
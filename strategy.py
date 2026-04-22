#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian breakout with 1d ADX trend filter and volume confirmation
    # Donchian(20) captures breakouts with clear entry/exit levels
    # 1d ADX > 25 filters for trending markets (works in bull/bear trends)
    # Volume spike (2x 20-period MA) confirms institutional participation
    # Designed for 6h timeframe to avoid excessive trading while capturing major moves
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX on 1d data
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_14_1d = adx  # Already smoothed
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Donchian Channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high + ADX > 25 (trending) + volume spike
            if close[i] > donchian_high[i] and adx_1d_aligned[i] > 25 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low + ADX > 25 (trending) + volume spike
            elif close[i] < donchian_low[i] and adx_1d_aligned[i] > 25 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to Donchian middle or trend weakness (ADX < 20)
            if position == 1:
                if close[i] < donchian_middle[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_middle[i] or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h ADX(14) trend filter + 1d volume spike
# - Long when price breaks above Donchian high(20) with volume > 2x 20-period 1d avg volume and 12h ADX > 25
# - Short when price breaks below Donchian low(20) with volume > 2x 20-period 1d avg volume and 12h ADX > 25
# - Exit when price returns to Donchian midpoint or ADX < 20 (trend weakening)
# - Designed to capture strong trending moves with institutional volume in 4h timeframe
# - Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
# - Works in both bull and bear markets by using ADX trend filter and volume confirmation

name = "4h_Donchian20_ADX12h_Volume1dSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for volume calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 12h data
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = df_12h['high'].diff()
    minus_dm = df_12h['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume spike filter: 1d volume > 2x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 2x 1d average volume (scaled)
        # Scale 1d average to 4h: 1d has 6x 4h bars, so divide by 6
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 2.0 * (vol_ma_1d_aligned[i] / 6.0)
        
        if position == 0:
            # Look for long entry: price breaks above Donchian high + volume spike + strong trend
            if (close[i] > donchian_high[i] and 
                close[i-1] <= donchian_high[i-1] and  # Just broke above
                volume_filter and 
                adx_12h_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below Donchian low + volume spike + strong trend
            elif (close[i] < donchian_low[i] and 
                  close[i-1] >= donchian_low[i-1] and  # Just broke below
                  volume_filter and 
                  adx_12h_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to midpoint or trend weakens
            if (close[i] <= donchian_mid[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to midpoint or trend weakens
            if (close[i] >= donchian_mid[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
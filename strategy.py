#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX regime filter
# Long when price breaks above Donchian(20) high + volume > 1.5x 20-period average + ADX > 25
# Short when price breaks below Donchian(20) low + volume > 1.5x 20-period average + ADX > 25
# Exit when price crosses Donchian(10) midpoint or ADX < 20 (range regime)
# Uses discrete position sizing (0.30) to balance capture and risk.
# Donchian channels provide clear structure, volume confirms breakout strength, ADX filters for trending markets.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading strong breakouts in trending conditions (ADX>25).

name = "4h_Donchian20_Volume_ADX_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d average volume for spike detection
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid_10 = (donchian_high_10 + donchian_low_10) / 2.0
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 20)  # Donchian20, ADX, and volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_vol_ma_20_1d = vol_ma_20_1d_aligned[i]
        curr_adx = adx[i]
        curr_donchian_high_20 = donchian_high_20[i]
        curr_donchian_low_20 = donchian_low_20[i]
        curr_donchian_mid_10 = donchian_mid_10[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below Donchian(10) midpoint OR ADX < 20 (range regime)
            if curr_close < curr_donchian_mid_10 or curr_adx < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian(10) midpoint OR ADX < 20 (range regime)
            if curr_close > curr_donchian_mid_10 or curr_adx < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Volume spike condition: current volume > 1.5x 1d average volume
            volume_spike = curr_volume > 1.5 * curr_vol_ma_20_1d
            
            # Long when price breaks above Donchian(20) high + volume spike + ADX > 25
            if curr_close > curr_donchian_high_20 and volume_spike and curr_adx > 25.0:
                signals[i] = 0.30
                position = 1
            # Short when price breaks below Donchian(20) low + volume spike + ADX > 25
            elif curr_close < curr_donchian_low_20 and volume_spike and curr_adx > 25.0:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
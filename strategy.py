#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dATR_VolumeSpike_TrendFilter_v1
Hypothesis: Trade 6h Donchian(20) breakouts with 1-day ATR volatility filter and volume spike confirmation. 
Only take breakouts when 1d ATR(14) is above its 50-period MA (high volatility regime) and 
volume > 2.0x 20-period average. Exit on opposite Donchian touch or when volatility regime ends. 
Works in bull (breakouts with high vol) and bear (breakdowns with high vol) markets. 
Position size: 0.25 to limit drawdown. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for ATR filter and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR MA
        return np.zeros(n)
    
    # Calculate 1-day ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) - Wilder's smoothing
    atr_14_1d = np.full_like(tr, np.nan, dtype=float)
    for i in range(len(tr)):
        if i == 0:
            atr_14_1d[i] = np.nan
        elif i < 14:
            if i == 1:
                atr_14_1d[i] = tr[i]
            else:
                valid_tr = tr[1:i+1][~np.isnan(tr[1:i+1])]
                if len(valid_tr) > 0:
                    atr_14_1d[i] = np.mean(valid_tr)
                else:
                    atr_14_1d[i] = np.nan
        else:
            if np.isnan(atr_14_1d[i-1]):
                atr_14_1d[i] = np.nan
            else:
                atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR(14) 50-period MA for volatility regime filter
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR and ATR MA to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    
    # Calculate 1-day Donchian(20) levels
    donch_high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20_1d)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR MA (50) and Donchian (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_ma_50_1d_aligned[i]) or
            np.isnan(donch_high_20_aligned[i]) or
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF volatility regime: ATR > ATR_MA (high volatility)
        high_vol_regime = atr_14_1d_aligned[i] > atr_ma_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above 1d Donchian(20) high + high vol + volume confirmation
            long_setup = (close[i] > donch_high_20_aligned[i]) and high_vol_regime and volume_confirm
            
            # Short setup: price breaks below 1d Donchian(20) low + high vol + volume confirmation
            short_setup = (close[i] < donch_low_20_aligned[i]) and high_vol_regime and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches 1d Donchian low OR volatility regime ends
            if (close[i] <= donch_low_20_aligned[i]) or (not high_vol_regime):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches 1d Donchian high OR volatility regime ends
            if (close[i] >= donch_high_20_aligned[i]) or (not high_vol_regime):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dATR_VolumeSpike_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0
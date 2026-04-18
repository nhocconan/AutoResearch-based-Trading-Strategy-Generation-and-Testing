#!/usr/bin/env python3
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
    
    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR for volatility filter (14-period)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = np.nan
    tr3_1d[0] = np.nan
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR MA (50-period) for volatility regime
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align 1d ATR and its MA to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Volatility filter: current ATR > 50-day average ATR (high volatility regime)
    vol_regime = atr_1d_aligned > atr_ma_50_aligned
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume spike (volume > 2.0x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 30) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_50_aligned[i]) or
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade in high volatility
        if not vol_regime[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike
            if close[i] > highest_high[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume spike
            elif close[i] < lowest_low[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if close[i] < lowest_low[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if close[i] > highest_high[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
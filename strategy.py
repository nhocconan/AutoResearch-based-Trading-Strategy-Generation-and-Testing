#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Donchian20_VolumeSpike_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min of last 20 daily highs/lows
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 6h ATR (14) for volatility filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 6h Volume Spike Filter ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_ratio_val = vol_ratio[i]
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(dch_high) or 
            np.isnan(dch_low) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and volatility filter
            if high_val > dch_high and vol_ratio_val > 2.5 and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and volatility filter
            elif low_val < dch_low and vol_ratio_val > 2.5 and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price closes below Donchian low OR volatility drops
            if close_val < dch_low or atr_val < atr[i-1] * 0.5:  # ATR contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price closes above Donchian high OR volatility drops
            if close_val > dch_high or atr_val < atr[i-1] * 0.5:  # ATR contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
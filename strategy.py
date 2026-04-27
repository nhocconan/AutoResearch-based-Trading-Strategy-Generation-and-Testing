# Solution for experiment 101162
#!/usr/bin/env python3
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
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR for volatility filtering
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_normalized = atr_1d / close_1d
    atr_1d_norm_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_normalized)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    # Price position within Donchian channel for mean reversion signals
    donchian_range = highest_high - lowest_low
    donchian_range_safe = np.where(donchian_range == 0, 1, donchian_range)
    price_position = (close - lowest_low) / donchian_range_safe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_norm_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: normalized ATR > 0.015 (1.5%)
        vol_filter = atr_1d_norm_aligned[i] > 0.015
        
        # Mean reversion conditions: extreme price positions + volume
        long_signal = (price_position[i] <= 0.1 and vol_filter and volume_filter[i])  # Near lower band
        short_signal = (price_position[i] >= 0.9 and vol_filter and volume_filter[i])  # Near upper band
        
        if long_signal and position != 1:
            signals[i] = 0.25
            position = 1
        elif short_signal and position != -1:
            signals[i] = -0.25
            position = -1
        # Exit conditions: return to middle of channel
        elif position == 1 and price_position[i] >= 0.5:
            signals[i] = 0.0
            position = 0
        elif position == -1 and price_position[i] <= 0.5:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_MeanReversion_VolumeVolFilter"
timeframe = "12h"
leverage = 1.0
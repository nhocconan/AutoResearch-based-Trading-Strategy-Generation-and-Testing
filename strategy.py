#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h price channel (Donchian 20) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6-period ATR for volatility filter (6h timeframe)
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=6, min_periods=6).mean().values
    
    # Align 6h data to primary timeframe (6h)
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === Daily Donchian channel (20-period) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # === Daily volume spike detection ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price_6h = close_6h[i // 4]  # 4x 6h bars in 1d, but we need current 6h close
        # Actually, we need the current 6h close price - use the 6h aligned close
        close_6h_aligned = align_htf_to_ltf(prices, df_6h, close_6h)
        price = close_6h_aligned[i]
        
        upper_level = upper_20_aligned[i]
        lower_level = lower_20_aligned[i]
        atr = atr_6h_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to midpoint of daily channel (mean reversion)
            midpoint = (upper_level + lower_level) / 2
            if price <= midpoint:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to midpoint of daily channel (mean reversion)
            midpoint = (upper_level + lower_level) / 2
            if price >= midpoint:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper Donchian with volume spike and volatility filter
            if price > upper_level and vol_spike and atr > 0:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below lower Donchian with volume spike and volatility filter
            elif price < lower_level and vol_spike and atr > 0:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_1dChannel_VolumeSpike_ATRFilter"
timeframe = "6h"
leverage = 1.0
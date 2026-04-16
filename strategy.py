#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 4h Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    donch_high = df_4h['high'].rolling(window=20, min_periods=20).max().values
    donch_low = df_4h['low'].rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === Volume Confirmation (4h volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are calculated
    warmup = max(20, 14) + 1
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Close position on opposite Donchian break or volatility contraction ===
        if position == 1:  # Long position
            # Exit when price breaks below Donchian low OR volatility drops too low
            if price < donch_low_aligned[i] or atr < 0.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above Donchian high OR volatility drops too low
            if price > donch_high_aligned[i] or atr < 0.5 * atr_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high with volume confirmation and sufficient volatility
            if price > donch_high_aligned[i] and vol_spike and atr > 0.5 * atr_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian low with volume confirmation and sufficient volatility
            elif price < donch_low_aligned[i] and vol_spike and atr > 0.5 * atr_1d_aligned[i]:
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

name = "4h_Donchian20_1dATR_VolumeFilter"
timeframe = "4h"
leverage = 1.0
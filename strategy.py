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
    
    # === 1d Donchian Channel (20 periods) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # === 1d Volume Spike (2.0x 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1d ADX (14 periods) for trend strength ===
    # Calculate directional movement
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - high_1d[i-1]), abs(low_1d[i] - low_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_14 = np.zeros(len(tr))
    plus_di_14 = np.zeros(len(plus_dm))
    minus_di_14 = np.zeros(len(minus_dm))
    
    atr_14[0] = tr[0]
    plus_di_14[0] = plus_dm[0]
    minus_di_14[0] = minus_dm[0]
    
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
        plus_di_14[i] = (plus_di_14[i-1] * 13 + plus_dm[i]) / 14
        minus_di_14[i] = (minus_di_14[i-1] * 13 + minus_dm[i]) / 14
    
    # Avoid division by zero
    dx = np.zeros(len(atr_14))
    mask = atr_14 > 0
    dx[mask] = 100 * np.abs(plus_di_14[mask] - minus_di_14[mask]) / (plus_di_14[mask] + minus_di_14[mask])
    
    # Calculate ADX (smoothed DX)
    adx_14 = np.zeros(len(dx))
    adx_14[0] = dx[0]
    for i in range(1, len(dx)):
        adx_14[i] = (adx_14[i-1] * 13 + dx[i]) / 14
    
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for calculations
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 2.0
        
        # ADX filter: only trade when trend is strong (ADX > 25)
        adx_strong = adx_14_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume confirmation and strong trend
            if close[i] > donchian_high_1d_aligned[i] and vol_confirm and adx_strong:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 1d Donchian low with volume confirmation and strong trend
            elif close[i] < donchian_low_1d_aligned[i] and vol_confirm and adx_strong:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to opposite Donchian level
        elif position == 1:
            # Exit long: price crosses below 1d Donchian low
            if close[i] < donchian_low_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d Donchian high
            if close[i] > donchian_high_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1dVolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0
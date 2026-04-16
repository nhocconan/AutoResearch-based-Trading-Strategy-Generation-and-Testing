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
    
    # === Daily data for ATR and volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === ATR(14) for volatility regime filter ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h Donchian channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (already aligned, but using for clarity)
    dh_4h = align_htf_to_ltf(prices, df_4h, donchian_high)
    dl_4h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Volatility regime filter: ATR > 50-day average of ATR ===
    atr_ma = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ma_4h = align_htf_to_ltf(prices, df_1d, atr_ma)
    high_volatility = atr_1d > atr_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    warmup = 200
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dh_4h[i]) or np.isnan(dl_4h[i]) or
            np.isnan(volume_spike[i]) or np.isnan(high_volatility[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh_level = dh_4h[i]
        dl_level = dl_4h[i]
        vol_spike = volume_spike[i]
        vol_regime = high_volatility[i]
        
        # === EXIT LOGIC: Exit when price crosses middle of channel or volatility drops ===
        if position == 1:  # Long position
            mid = (dh_level + dl_level) / 2
            if price < mid or not vol_regime:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            mid = (dh_level + dl_level) / 2
            if price > mid or not vol_regime:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high with volume spike and high volatility
            if price > dh_level and vol_spike and vol_regime:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below Donchian low with volume spike and high volatility
            elif price < dl_level and vol_spike and vol_regime:
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

name = "4h_Donchian20_VolumeSpike_VolatilityRegime"
timeframe = "4h"
leverage = 1.0
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
    
    # === 1h data (primary) ===
    # === 4h data (HTF for trend) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # === 1d data (HTF for volume regime) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 4h EMA21 for trend direction ===
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # === 1d volume regime (high/low volume days) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1h volume confirmation ===
    vol_ma_10_1h = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1h = volume / vol_ma_10_1h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 30
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(vol_ratio_1h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_21 = ema_21_4h_aligned[i]
        vol_regime = vol_ratio_1d_aligned[i]  # >1 = high volume day, <1 = low volume day
        vol_1h = vol_ratio_1h[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below EMA21
            if price < ema_21:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above EMA21
            if price > ema_21:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during high volume days (institutional participation)
            # AND with 1h volume confirmation
            if vol_regime > 1.2 and vol_1h > 1.3:
                # LONG: price above EMA21 (uptrend)
                if price > ema_21:
                    signals[i] = 0.20
                    position = 1
                    continue
                # SHORT: price below EMA21 (downtrend)
                elif price < ema_21:
                    signals[i] = -0.20
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA21_VolumeRegime_Filter"
timeframe = "1h"
leverage = 1.0
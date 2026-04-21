#!/usr/bin/env python3
"""
12h_HTF_Donchian20_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 1d Donchian(20) breakout + 12h volume spike (>1.5x 20-bar MA) for entry + ATR(14) stoploss (1.5x). 
Add regime filter: only trade when 1d ADX(14) > 25 to avoid choppy markets. 
Target 12-30 trades/year per symbol. Works in bull (breakouts) and bear (tight stops limit losses) via volume/ADX confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for daily Donchian channels and ADX
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high/low (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1d ADX (14-period) for regime filter ===
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    # Smooth TR and DM
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr1_12h[0] = tr2_12h[0] = tr3_12h[0] = np.nan
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        adx_ok = adx_aligned[i] > 25.0  # regime filter: only trade when strongly trending
        
        if position == 0:
            # Long: break above 1d Donchian high with volume spike and ADX > 25
            if price > donchian_high_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below 1d Donchian low with volume spike and ADX > 25
            elif price < donchian_low_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < donchian_high_aligned[i-1] - 1.5 * atr[i] or (price < donchian_low_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > donchian_low_aligned[i-1] + 1.5 * atr[i] or (price > donchian_high_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_Donchian20_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
4h_HTF_Camarilla_R3S3_Breakout_VolumeSpike_ATRStop_V2
Hypothesis: Use 1d Camarilla R3/S3 levels + 4h volume spike (>2x 20-bar MA) for breakout entry + ATR(14) stoploss (2.0x). 
Adds regime filter: only trade when 4h ADX(14) > 20 to avoid choppy markets. 
Target 15-25 trades/year per symbol. Works in bull (breakouts) and bear (tight stops limit losses) via volume/ADX confluence.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for daily Camarilla pivots
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    camarilla_r3 = pivot + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3 = pivot - 1.1 * (high_1d - low_1d) / 2.0
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ADX (14-period) for regime filter
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        adx_ok = adx[i] > 20.0  # regime filter: only trade when trending
        
        if position == 0:
            # Long: break above Camarilla R3 with volume spike and ADX > 20
            if price > camarilla_r3_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S3 with volume spike and ADX > 20
            elif price < camarilla_s3_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < camarilla_r3_aligned[i-1] - 2.0 * atr[i] or (price < camarilla_s3_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > camarilla_s3_aligned[i-1] + 2.0 * atr[i] or (price > camarilla_r3_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Camarilla_R3S3_Breakout_VolumeSpike_ATRStop_V2"
timeframe = "4h"
leverage = 1.0
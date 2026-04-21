#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_R1S1_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 4h Camarilla R1/S1 levels + 1h volume spike (>2x 20-bar MA) for breakout entry + ATR(14) stop (2.0x).
Add 4h ADX(14) > 20 regime filter to reduce whipsaw. Session filter: trade only 08-20 UTC.
Target 15-35 trades/year per symbol. Works in bull/bear via volatility expansion + trend filter.
Timeframe: 1h, HTF: 4h for direction/levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')  # for 4h Camarilla pivots and ADX
    
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # === 4h Camarilla Pivot Levels (R1, S1) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot point
    pivot = (high_4h + low_4h + close_4h) / 3.0
    # Camarilla levels
    camarilla_r1 = pivot + 1.1 * (high_4h - low_4h) / 4.0
    camarilla_s1 = pivot - 1.1 * (high_4h - low_4h) / 4.0
    
    # Align to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # === 4h ADX (14-period) for regime filter ===
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    plus_dm = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # === 1h Indicators ===
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
    
    # Session filter: 08-20 UTC (precomputed for speed)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if not in session or indicators not ready
        if not in_session[i] or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        adx_ok = adx_aligned[i] > 20.0  # regime filter: only trade when ADX > 20
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and ADX > 20
            if price > camarilla_r1_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla S1 with volume spike and ADX > 20
            elif price < camarilla_s1_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < camarilla_r1_aligned[i-1] - 2.0 * atr[i] or \
               (price < camarilla_s1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > camarilla_s1_aligned[i-1] + 2.0 * atr[i] or \
               (price > camarilla_r1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_Pivot_R1S1_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "1h"
leverage = 1.0
#!/usr/bin/env python3
"""
1d_HTF_Camarilla_R1S1_Breakout_VolumeATRStop_V1
Hypothesis: Use 1w Camarilla R1/S1 levels + 1d volume spike (>1.8x 20-bar MA) for breakout entry + ATR(14) stoploss (1.8x). 
Add regime filter: only trade when 1d ADX(14) > 22 to avoid choppy markets. 
Target 8-18 trades/year per symbol. Works in bull (breakouts) and bear (tight stops limit losses) via volume/ADX confluence.
Primary timeframe: 1d, HTF: 1w. Designed for BTC/ETH resilience in ranging/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for weekly Camarilla pivots
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === 1w Camarilla Pivot Levels (R1, S1) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    pivot = (high_1w + low_1w + close_1w) / 3.0
    # Camarilla levels (R1, S1)
    camarilla_r1 = pivot + 1.1 * (high_1w - low_1w) / 4.0
    camarilla_s1 = pivot - 1.1 * (high_1w - low_1w) / 4.0
    
    # Align to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # === 1d Indicators ===
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
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.8 * vol_ma[i]  # volume spike confirmation
        adx_ok = adx[i] > 22.0  # regime filter: only trade when trending
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and ADX > 22
            if price > camarilla_r1_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and ADX > 22
            elif price < camarilla_s1_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < camarilla_r1_aligned[i-1] - 1.8 * atr[i] or (price < camarilla_s1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > camarilla_s1_aligned[i-1] + 1.8 * atr[i] or (price > camarilla_r1_aligned[i-1] and vol_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_Camarilla_R1S1_Breakout_VolumeATRStop_V1"
timeframe = "1d"
leverage = 1.0
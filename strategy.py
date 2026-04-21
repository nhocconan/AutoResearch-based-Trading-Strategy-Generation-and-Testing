#!/usr/bin/env python3
"""
4h_HTF_1d_Donchian20_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use 1d Donchian(20) channels + 4h volume spike (>2x 20-bar MA) for breakout entry + ATR(14) stoploss (2.0x). 
Add regime filter: only trade when 4h ADX(14) > 25 (strong trend filter) to reduce whipsaw in ranging markets. 
Uses discrete position sizing (0.30) to balance return and drawdown. Target 20-40 trades/year per symbol. 
Works in bull (breakouts capture momentum) and bear (tight stops limit losses during reversals).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')  # for 1d Donchian channels
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Donchian Channels (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
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
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume spike confirmation
        adx_ok = adx[i] > 25.0  # strong trend filter: only trade when sufficient trend
        
        if position == 0:
            # Long: break above 1d Donchian high with volume spike and ADX > 25
            if price > donch_high_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = 0.30
                position = 1
            # Short: break below 1d Donchian low with volume spike and ADX > 25
            elif price < donch_low_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < donch_high_aligned[i-1] - 2.0 * atr[i] or price < donch_low_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > donch_low_aligned[i-1] + 2.0 * atr[i] or price > donch_high_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_HTF_1d_Donchian20_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0
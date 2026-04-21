#!/usr/bin/env python3
"""
4h_HTF_1w_Donchian20_Breakout_VolumeSpike_ATRStop_V1
Hypothesis: Use weekly Donchian(20) channel breakouts on 4h chart with volume spike (>1.5x 50-bar MA) and ATR(14) stoploss (2.0x). 
Add regime filter: only trade when 4h ADX(14) > 20 (moderate trend) to avoid choppy markets. 
Uses discrete position sizing (0.25) to minimize fee drag while capturing momentum. 
Weekly structure provides strong support/resistance that works in both bull (breakouts) and bear (mean reversion at extremes) markets. 
Target 15-30 trades/year per symbol to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')  # for weekly Donchian channels
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Donchian Channel (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high of last 20 weekly candles
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 weekly candles
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # === 4h Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (50-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
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
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume spike confirmation
        adx_ok = adx[i] > 20.0  # moderate trend filter
        
        if position == 0:
            # Long: break above weekly Donchian upper with volume spike and ADX > 20
            if price > upper_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian lower with volume spike and ADX > 20
            elif price < lower_aligned[i-1] and vol_ok and adx_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: ATR stoploss or opposite signal
            if price < upper_aligned[i-1] - 2.0 * atr[i] or price < lower_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: ATR stoploss or opposite signal
            if price > lower_aligned[i-1] + 2.0 * atr[i] or price > upper_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_1w_Donchian20_Breakout_VolumeSpike_ATRStop_V1"
timeframe = "4h"
leverage = 1.0
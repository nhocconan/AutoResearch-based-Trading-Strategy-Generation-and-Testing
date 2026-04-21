#!/usr/bin/env python3
"""
12h_HTF_1d_Donchian20_VolumeSpike_ATRStop_V1
Hypothesis: 12h Donchian(20) breakout with 1d HTF volume confirmation (>1.5x 20-period volume MA) and ATR-based stoploss. 
Primary timeframe 12h reduces trade frequency to avoid fee drag. Volume spike filters false breakouts. 
ATR stoploss manages risk. Target 12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by capturing breakouts in direction of 1d trend (price > 1d EMA50 for longs, < for shorts).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for volume and EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Volume MA for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_12h - low_12h).values
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1))).values
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1))).values
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol_12h = volume_12h[i]
        vol_ok = vol_12h > 1.5 * vol_ma_1d_aligned[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + 1d uptrend
            if price > highest_high[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr[i]
            # Short: price breaks below Donchian lower + volume confirmation + 1d downtrend
            elif price < lowest_low[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr[i]
        
        elif position == 1:
            # Update ATR trailing stop for long
            atr_stop = max(atr_stop, price - 2.0 * atr[i])
            # Exit long: price breaks below ATR stop or Donchian lower
            if price < atr_stop or price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update ATR trailing stop for short
            atr_stop = min(atr_stop, price + 2.0 * atr[i])
            # Exit short: price breaks above ATR stop or Donchian upper
            if price > atr_stop or price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1d_Donchian20_VolumeSpike_ATRStop_V1"
timeframe = "12h"
leverage = 1.0
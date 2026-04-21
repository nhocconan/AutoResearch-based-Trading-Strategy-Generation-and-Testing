#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1
Hypothesis: 4h Donchian(20) breakout with 12h trend filter (price > 12h EMA34 for longs, < for shorts) and volume confirmation (>1.5x 20-period volume MA). ATR-based stoploss (2.0x ATR) manages risk. 
Donchian channels capture volatility breakouts; 12h EMA34 filters for higher-timeframe trend alignment. 
Volume confirmation reduces false breakouts. Target 20-50 trades/year (80-200 total over 4 years).
Uses 4h primary timeframe with 12h HTF for EMA trend and ATR calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (12h for EMA trend and ATR)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h EMA34 for trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h ATR(14) for stoploss ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_12h[1:] - close_12h[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # first TR is NaN
    atr_14_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) 
            or np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i])
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        atr = atr_14_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + 12h uptrend
            if price > highest_20[i] and vol_ok and price > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower + volume confirmation + 12h downtrend
            elif price < lowest_20[i] and vol_ok and price < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Update signal to hold position
            signals[i] = 0.25
            # Stoploss: price drops below entry - 2.0 * ATR
            if price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Update signal to hold position
            signals[i] = -0.25
            # Stoploss: price rises above entry + 2.0 * ATR
            if price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_ATRStop_V1"
timeframe = "4h"
leverage = 1.0
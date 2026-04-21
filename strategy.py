#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_ATRStop_v1
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume confirmation (>1.5x 20-period average volume) capture institutional momentum. ATR-based stoploss (2.5x ATR) controls drawdown. Works in both bull (breakouts up) and bear (breakouts down) markets. Target: 100-180 total trades over 4 years (25-45/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high/low (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR for stoploss and volatility filter ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d EMA50 for trend filter (avoid counter-trend trades) ===
    if len(df_1d) >= 50:
        ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema_50 = ema_50_aligned[i]
        
        # Volume spike condition: >1.5x 20-period average volume
        volume_spike = vol > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + above 1d EMA50
            if price > donchian_high[i] and volume_spike and price > ema_50:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_at_entry = atr_val
            # Short: price breaks below Donchian low + volume spike + below 1d EMA50
            elif price < donchian_low[i] and volume_spike and price < ema_50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_at_entry = atr_val
        
        elif position != 0:
            # ATR-based stoploss: 2.5x ATR from entry
            stop_distance = 2.5 * atr_at_entry
            
            if position == 1:
                # Long: exit if price drops below stop or re-enters Donchian channel
                if price < entry_price - stop_distance or price < donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short: exit if price rises above stop or re-enters Donchian channel
                if price > entry_price + stop_distance or price > donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0
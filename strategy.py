#!/usr/bin/env python3
"""
1d_HTF_1w_Donchian20_VolumeSpike_ATRStop_V1
Hypothesis: Use 1d primary timeframe with 1w Donchian channel breakout for strong momentum capture in both bull and bear markets.
Add volume confirmation (>2.0x 20-bar volume MA) and ATR-based stoploss to reduce whipsaw and manage risk.
Position size 0.25 balances risk/return. Target 15-25 trades/year per symbol.
Works in bull via upward breakouts, bear via downward breakouts, with ATR stop protecting against reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1w Donchian Channel (20-period) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian bands: highest high and lowest low over 20 periods
    highest_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # === 1d Indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 2.0 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper band + volume confirmation
            if price > highest_high_aligned[i-1] and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1w Donchian lower band + volume confirmation
            elif price < lowest_low_aligned[i-1] and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # ATR-based stoploss: exit if price drops below entry - 2.5 * ATR
            stop_price = entry_price - 2.5 * atr[i]
            if price < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # ATR-based stoploss: exit if price rises above entry + 2.5 * ATR
            stop_price = entry_price + 2.5 * atr[i]
            if price > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_HTF_1w_Donchian20_VolumeSpike_ATRStop_V1"
timeframe = "1d"
leverage = 1.0
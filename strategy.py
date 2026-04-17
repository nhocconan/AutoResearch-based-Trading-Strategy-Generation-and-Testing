#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
- Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period volume MA
- Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period volume MA
- Exit on ATR(14) trailing stop: long exits when price < highest_high_since_entry - 2.5*ATR
- Fixed position size 0.25 to balance reward/risk and limit fee drag
- Designed for 4h timeframe with strict entry conditions to target 75-200 trades over 4 years
- Works in bull markets (buying breakouts) and bear markets (selling breakdowns)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period Donchian high/low on 4h
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 4h for confirmation
    volume_4h = df_4h['volume'].values
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) on 4h for stoploss calculation
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF indicators to primary timeframe (4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for Donchian(20) + ATR(14)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        dch_high = donchian_high_aligned[i]
        dch_low = donchian_low_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation
            # Long: price breaks above Donchian high + volume spike
            if price > dch_high and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
                highest_since_entry = high_price
            # Short: price breaks below Donchian low + volume spike
            elif price < dch_low and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = low_price
        
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high_price)
            # ATR trailing stop: exit when price drops below highest - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low_price)
            # ATR trailing stop: exit when price rises above lowest + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Donchian channel breakout with volume confirmation and ATR-based trailing stop.
- Long when price breaks above 1d Donchian(20) high + volume > 2.0x 4h volume MA(20)
- Short when price breaks below 1d Donchian(20) low + volume > 2.0x 4h volume MA(20)
- Exit on ATR trailing stop (3x ATR(14)) from extreme price
- Fixed position size 0.30 to balance return and drawdown
- Uses multi-timeframe structure (1d for trend/channel, 4h for execution) with volume confirmation to filter false breakouts
- Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year)
- ATR trailing stop adapts to volatility, effective in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for volume MA (primary timeframe execution)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Donchian channel and ATR (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20-period)
    donchian_high_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for trailing stop
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF indicators to primary timeframe (4h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    max_high_since_entry = 0.0  # for long trailing stop
    min_low_since_entry = 0.0   # for short trailing stop
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        atr_val = atr_14_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation
            # Long: price breaks above 1d Donchian high + volume spike
            if price > donchian_high and vol > 2.0 * vol_ma:
                signals[i] = 0.30
                position = 1
                entry_price = price
                max_high_since_entry = price
            # Short: price breaks below 1d Donchian low + volume spike
            elif price < donchian_low and vol > 2.0 * vol_ma:
                signals[i] = -0.30
                position = -1
                entry_price = price
                min_low_since_entry = price
        
        elif position == 1:
            # Update highest price since entry for trailing stop
            max_high_since_entry = max(max_high_since_entry, price)
            # ATR trailing stop: exit if price drops 3*ATR from peak
            if price < max_high_since_entry - 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Update lowest price since entry for trailing stop
            min_low_since_entry = min(min_low_since_entry, price)
            # ATR trailing stop: exit if price rises 3*ATR from trough
            if price > min_low_since_entry + 3.0 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Donchian channel breakout with volume confirmation and ATR-based stop.
- Long when price breaks above 1w Donchian high(20) + volume > 2.0x 20-period 1d volume MA
- Short when price breaks below 1w Donchian low(20) + volume > 2.0x 20-period 1d volume MA
- Exit on opposite Donchian break or ATR trailing stop (2.5 * ATR)
- Fixed position size 0.25 to manage drawdown
- Uses weekly structure for major trend, daily execution for timing, volume confirmation to filter noise
- Designed for 1d timeframe with strict entry conditions to limit trades to 30-100 total over 4 years
- Donchian breakout captures strong momentum moves, effective in both accumulation and distribution phases
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
    
    # Get 1d data for volume MA and ATR
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Volume average (20-period) on 1d for confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Get 1w data for Donchian channel (HTF for structure)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channel (20-period)
    donch_high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align all HTF indicators to primary timeframe (1d)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20_1w)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        atr = atr_14_aligned[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation
            # Long: price breaks above 1w Donchian high + volume spike
            if price > donch_high and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below 1w Donchian low + volume spike
            elif price < donch_low and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, price)
            
            # Exit on opposite Donchian break or ATR trailing stop
            if price < donch_low or price < highest_since_entry - 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit on opposite Donchian break or ATR trailing stop
            if price > donch_high or price > lowest_since_entry + 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1w_VolumeSpike_ATRTrail"
timeframe = "1d"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Donchian upper (20-period high) + volume > 1.5x 20-period 4h volume MA + price above 1d EMA34
- Short when price breaks below Donchian lower (20-period low) + volume > 1.5x 20-period 4h volume MA + price below 1d EMA34
- Fixed position size 0.25 to manage drawdown
- Uses Donchian breakout for structure, volume confirmation for conviction, 1d EMA34 for multi-timeframe trend alignment
- Designed for 4h timeframe with moderate entry frequency (~25-40 trades/year) to avoid fee drag
- Works in both bull (breakouts with trend) and bear (breakouts against trend filtered by 1d EMA) markets
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
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter (HTF for structure)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian channels (20-period) on primary timeframe (4h)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align all HTF indicators to primary timeframe (4h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        ema_34_val = ema_34_aligned[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and 1d EMA34 trend filter
            # Long: price breaks above Donchian upper + volume spike + price above 1d EMA34
            if price > upper_donchian and vol > 1.5 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower + volume spike + price below 1d EMA34
            elif price < lower_donchian and vol > 1.5 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1d EMA34 (trend change) or Donchian lower (mean reversion)
            if price < ema_34_val or price < lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1d EMA34 (trend change) or Donchian upper (mean reversion)
            if price > ema_34_val or price > upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_1dEMA34"
timeframe = "4h"
leverage = 1.0
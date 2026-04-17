#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian(20) upper channel + price above 12h EMA34 + volume > 1.5x 20-period 4h volume MA.
Short when price breaks below Donchian(20) lower channel + price below 12h EMA34 + volume > 1.5x 20-period 4h volume MA.
Exit on opposite Donchian break or 12h EMA34 cross.
Fixed position size 0.25 to manage drawdown. Designed for 4h timeframe with tight entry conditions
to target 75-200 total trades over 4 years (19-50/year). Uses price channel breakout structure
which works in both bull (breakouts) and bear (breakdowns) markets, with volume confirmation
reducing false signals and 12h EMA34 ensuring trend alignment.
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
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter (HTF for structure)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all HTF indicators to primary timeframe (4h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        ema_34_val = ema_34_aligned[i]
        upper_donchian = highest_high_20[i]
        lower_donchian = lowest_low_20[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and 12h EMA34 trend filter
            # Long: price breaks above Donchian upper + volume spike + price above 12h EMA34
            if price > upper_donchian and vol > 1.5 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower + volume spike + price below 12h EMA34
            elif price < lower_donchian and vol > 1.5 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below Donchian lower or 12h EMA34 (trend change)
            if price < lower_donchian or price < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above Donchian upper or 12h EMA34 (trend change)
            if price > upper_donchian or price > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_12hEMA34"
timeframe = "4h"
leverage = 1.0
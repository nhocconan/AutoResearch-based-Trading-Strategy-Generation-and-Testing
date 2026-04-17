#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and 1w EMA34 trend filter.
- Long when price breaks above 1d Donchian upper (20) + volume > 2.0x 20-period 12h volume MA + price above 1w EMA34
- Short when price breaks below 1d Donchian lower (20) + volume > 2.0x 20-period 12h volume MA + price below 1w EMA34
- Fixed position size 0.25 to manage drawdown
- Uses price channel breakout structure (works in trending markets) + volume confirmation + HTF trend filter
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
- Donchian breakout captures strong moves, effective in both bull and bear markets when aligned with weekly trend
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
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Volume average (20-period) on 12h for confirmation
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Donchian channel (20)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channel (20-period)
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all HTF indicators to primary timeframe (12h)
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_ma = volume_ma_20_aligned[i]
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and 1w EMA34 trend filter
            # Long: price breaks above 1d Donchian upper + volume spike + price above 1w EMA34
            if price > donchian_upper and vol > 2.0 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below 1d Donchian lower + volume spike + price below 1w EMA34
            elif price < donchian_lower and vol > 2.0 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 1w EMA34 (trend change) or opposite Donchian level
            if price < ema_34_val or price < donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 1w EMA34 (trend change) or opposite Donchian level
            if price > ema_34_val or price > donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_1wEMA34"
timeframe = "12h"
leverage = 1.0
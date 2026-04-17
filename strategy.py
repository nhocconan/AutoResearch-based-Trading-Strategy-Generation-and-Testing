#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and 12h EMA34 trend filter.
- Long when price breaks above Donchian(20) upper band + volume > 2.0x 20-period 4h volume MA + price above 12h EMA34
- Short when price breaks below Donchian(20) lower band + volume > 2.0x 20-period 4h volume MA + price below 12h EMA34
- Fixed position size 0.25 to manage drawdown
- Uses Donchian breakout for momentum + volume confirmation + 12h EMA trend
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
- Works in bull markets (buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend)
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
    
    # Donchian(20) on primary timeframe (4h)
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter (HTF for structure)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to primary timeframe (4h)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_roll_20[i]) or np.isnan(low_roll_20[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        donchian_upper = high_roll_20[i]
        donchian_lower = low_roll_20[i]
        vol_ma = volume_ma_20[i]
        ema_34_val = ema_34_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 12h EMA34 trend filter
            # Long: price breaks above Donchian upper + volume spike + price above 12h EMA34
            if price > donchian_upper and vol > 2.0 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian lower + volume spike + price below 12h EMA34
            elif price < donchian_lower and vol > 2.0 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit on close below 12h EMA34 (trend change) or opposite Donchian level
            if price < ema_34_val or price < donchian_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on close above 12h EMA34 (trend change) or opposite Donchian level
            if price > ema_34_val or price > donchian_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_12hEMA34"
timeframe = "4h"
leverage = 1.0
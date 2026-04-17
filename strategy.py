#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian channel breakout with volume confirmation and 12h EMA trend filter.
- Enter long when price breaks above Donchian(20) high + volume > 1.5x 20-period volume MA + price above 12h EMA(34)
- Enter short when price breaks below Donchian(20) low + volume > 1.5x 20-period volume MA + price below 12h EMA(34)
- Exit when price crosses back inside Donchian channel
- Fixed position size 0.25 to manage drawdown
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
- Combines price breakout with volume confirmation and higher timeframe trend alignment
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
    
    # Donchian Channel (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Load 12h EMA(34) once before loop
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # warmup for 12h EMA(34)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high.iloc[i]) or np.isnan(donchian_low.iloc[i]) or 
            np.isnan(volume_ma_20.iloc[i]) or np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = donchian_high.iloc[i]
        lower = donchian_low.iloc[i]
        ema_val = ema_34_12h_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and 12h trend filter
            # Long: price breaks above Donchian high + volume spike + price above 12h EMA
            if price > upper and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + price below 12h EMA
            elif price < lower and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back inside Donchian channel
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back inside Donchian channel
            if price < upper and price > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_12hEMA34"
timeframe = "4h"
leverage = 1.0
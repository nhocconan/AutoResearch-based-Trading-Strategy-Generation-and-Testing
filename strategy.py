#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 12h EMA34 trend filter and volume confirmation.
- Long when price breaks above Donchian(20) high and price > 12h EMA34 with volume > 1.5x 20-period volume MA
- Short when price breaks below Donchian(20) low and price < 12h EMA34 with volume > 1.5x 20-period volume MA
- Exit when price crosses back through Donchian middle line (mean of 20-period high/low)
- Position size 0.25 to manage drawdown
- Designed for 4h timeframe with tight entry conditions to limit trades to 75-200 total over 4 years
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
    
    # Get 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = high_20[i]
        lower = low_20[i]
        middle = donchian_mid[i]
        ema_val = ema_34_aligned[i]
        
        if position == 0:
            # Look for breakout with volume confirmation and trend filter
            # Long: break above upper band, price above EMA34, volume spike
            if price > upper and price > ema_val and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band, price below EMA34, volume spike
            elif price < lower and price < ema_val and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses back below middle line
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses back above middle line
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0
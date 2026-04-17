#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian channel breakout with 12h EMA trend filter and volume spike.
- Calculate Donchian(20) from previous 12h OHLC: upper = 20-period high, lower = 20-period low
- Enter long when price breaks above upper band with volume > 1.8x 20-period volume MA and price above 12h EMA34
- Enter short when price breaks below lower band with volume > 1.8x 20-period volume MA and price below 12h EMA34
- Exit when price crosses back to the opposite band (lower for longs, upper for shorts)
- Fixed position size 0.25 to manage drawdown
- Uses 12h trend filter to avoid counter-trend trades
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
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
    
    # Get 12-hour data for Donchian calculation and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian(20) from previous 12h OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band = 20-period high, Lower band = 20-period low
    upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (use previous 12h values)
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_val = ema_34_aligned[i]
        
        if position == 0:
            # Look for Donchian breakout with volume confirmation and trend filter
            # Long: price breaks above upper band + volume spike + price above 12h EMA34
            if price > upper_val and vol > 1.8 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume spike + price below 12h EMA34
            elif price < lower_val and vol > 1.8 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below lower band (opposite level)
            if price < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above upper band (opposite level)
            if price > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_12hEMA34"
timeframe = "4h"
leverage = 1.0
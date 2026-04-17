#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using Donchian(20) breakout with 1d EMA100 trend filter and volume confirmation.
- Calculate Donchian channels (20-period high/low) on 12h data
- Enter long when price breaks above 20-period high with volume > 1.5x 20-period volume MA and price above 1d EMA100
- Enter short when price breaks below 20-period low with volume > 1.5x 20-period volume MA and price below 1d EMA100
- Exit when price crosses back to the opposite 20-period level (20-period low for shorts, 20-period high for longs)
- Fixed position size 0.25 to manage drawdown
- Uses 1d trend filter to avoid counter-trend trades
- Designed for 12h timeframe with strict entry conditions to limit trades to 50-150 total over 4 years
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
    
    # Get 1-day data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA100 for trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # Calculate 20-period Donchian channels on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20.iloc[i]) or np.isnan(ema_100_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        ema_val = ema_100_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and trend filter
            # Long: price breaks above 20-period high + volume spike + price above 1d EMA100
            if price > dh and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low + volume spike + price below 1d EMA100
            elif price < dl and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below 20-period low (opposite level)
            if price < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above 20-period high (opposite level)
            if price > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_Volume_1dEMA100"
timeframe = "12h"
leverage = 1.0
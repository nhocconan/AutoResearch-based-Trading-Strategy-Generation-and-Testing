#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian channel breakout with 1d EMA trend filter and volume spike.
- Calculate Donchian upper/lower bands from previous 20 bars (20-period high/low)
- Enter long when price breaks above upper band with volume > 1.5x 20-period volume MA and price above 1d EMA50
- Enter short when price breaks below lower band with volume > 1.5x 20-period volume MA and price below 1d EMA50
- Exit when price crosses back to the opposite Donchian band (lower band for shorts, upper band for longs)
- Fixed position size 0.25 to manage drawdown
- Uses 1d trend filter to avoid counter-trend trades
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
    
    # Get 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from previous 20 periods (high/low of last 20 bars)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_ma_20.iloc[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and trend filter
            # Long: price breaks above upper band + volume spike + price above 1d EMA50
            if price > upper_band and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume spike + price below 1d EMA50
            elif price < lower_band and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below lower band (opposite band)
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above upper band (opposite band)
            if price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_1dEMA50"
timeframe = "4h"
leverage = 1.0
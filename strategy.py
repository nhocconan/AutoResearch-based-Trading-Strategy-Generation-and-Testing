#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using Weekly Donchian breakout with volume confirmation and 1w EMA trend filter.
- Calculate Donchian channels from previous week: upper (20-period high), lower (20-period low)
- Enter long when price breaks above upper band with volume > 1.5x 20-period volume MA and price above 1w EMA20
- Enter short when price breaks below lower band with volume > 1.5x 20-period volume MA and price below 1w EMA20
- Exit when price crosses back to the opposite band (lower for longs, upper for shorts)
- Fixed position size 0.25 to manage drawdown
- Uses 1w trend filter to avoid counter-trend trades
- Designed for 1d timeframe with strict entry conditions to limit trades to 30-100 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian channels from previous week's high/low (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: 20-period high
    upper_series = pd.Series(high_1w).rolling(window=20, min_periods=20).max()
    upper = upper_series.values
    # Lower band: 20-period low
    lower_series = pd.Series(low_1w).rolling(window=20, min_periods=20).min()
    lower = lower_series.values
    
    # Align Donchian levels to daily timeframe (use previous week's levels)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    
    # Volume confirmation: 20-period volume MA
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_val = ema_20_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and trend filter
            # Long: price breaks above upper band + volume spike + price above 1w EMA20
            if price > upper_val and vol > 1.5 * vol_ma and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume spike + price below 1w EMA20
            elif price < lower_val and vol > 1.5 * vol_ma and price < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below lower band (opposite band)
            if price < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above upper band (opposite band)
            if price > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchianBreakout_Volume_1wEMA20"
timeframe = "1d"
leverage = 1.0
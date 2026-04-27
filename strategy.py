#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 12-hour volume confirmation and 12-hour trend filter.
Trades breakouts when price closes above/below Donchian(20) with volume above 12h average and 12h EMA confirms direction.
Designed to work in both bull and bear markets by using 12h trend filter and volume to confirm breakout strength.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
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
    
    # Get 4-hour data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channel (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4-hour timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12-hour data for volume filter and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 12-hour EMA(25) for trend
    close_12h = df_12h['close'].values
    ema_25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_25_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian levels, volume MA, and 12h EMA
    start_idx = max(20, 20, 25)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or np.isnan(ema_25_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current 4-hour price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        trend_12h = ema_25_12h_aligned[i]
        
        # Current Donchian levels
        upper_band = donchian_high_aligned[i]
        lower_band = donchian_low_aligned[i]
        
        # Volume filter: volume > 1.5x 12-hour average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: Donchian breakout with volume and 12h trend alignment
        if position == 0:
            # Long: price breaks above upper band with volume + 12h uptrend
            if price_now > upper_band and vol_filter and price_now > trend_12h:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume + 12h downtrend
            elif price_now < lower_band and vol_filter and price_now < trend_12h:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midline or 12h trend turns down
            midline = (upper_band + lower_band) / 2
            if price_now < midline or price_now < trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midline or 12h trend turns up
            midline = (upper_band + lower_band) / 2
            if price_now > midline or price_now > trend_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_12hVolume_12hTrend"
timeframe = "4h"
leverage = 1.0
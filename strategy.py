#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and 1-day ATR volatility filter.
Trades breakouts of the 20-period Donchian channel when volume exceeds 1.5x the 20-day average and ATR(14) is below its 50-day median (low volatility environment).
Designed to work in both bull and bear markets by using volatility filter to avoid whipsaws and volume to confirm breakout strength.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
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
    
    # Get 12-hour data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12-hour Donchian channels (20-period high/low)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get daily data for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1-day ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-day median of ATR for volatility regime filter
    atr_median_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).median().values
    atr_median_50_aligned = align_htf_to_ltf(prices, df_1d, atr_median_50)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, volume MA, ATR and its median
    start_idx = max(20, 20, 14, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_median_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_now = atr_14_aligned[i]
        atr_med = atr_median_50_aligned[i]
        
        # Current Donchian levels
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        
        # Volume filter: volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Volatility filter: ATR < median ATR (low volatility environment)
        vol_regime_filter = atr_now < atr_med
        
        # Entry conditions: Donchian breakout with volume and low volatility
        if position == 0:
            # Long: price breaks above Donchian high with volume + low vol
            if price_now > donch_high and vol_filter and vol_regime_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume + low vol
            elif price_now < donch_low and vol_filter and vol_regime_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or volatility increases
            if price_now < donch_low or not vol_regime_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high or volatility increases
            if price_now > donch_high or not vol_regime_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_1dVolume_VolatilityFilter"
timeframe = "12h"
leverage = 1.0
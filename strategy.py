#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian(20) breakout with daily volume confirmation and weekly ADX trend filter.
Enters long when price breaks above 20-period Donchian high with volume > 1.5x daily average and weekly ADX > 25.
Enters short when price breaks below 20-period Donchian low with volume > 1.5x daily average and weekly ADX > 25.
Uses weekly ADX for trend strength to avoid ranging markets. Position size 0.25 to limit drawdown.
Target: 20-30 trades/year per symbol (80-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    dm_plus_period = pd.Series(dm_plus).rolling(window=period, min_periods=period).sum().values
    dm_minus_period = pd.Series(dm_minus).rolling(window=period, min_periods=period).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_period / tr_period
    minus_di = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_14_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_14_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, volume MA, and weekly ADX
    start_idx = max(20, 20, 20, 20, 14 + 14)  # Donchian(20), Vol MA(20), ADX needs 2*period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        adx_1w = adx_14_1w_aligned[i]
        
        # Current Donchian levels
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Trend filter: weekly ADX > 25 (trending market)
        trend_filter = adx_1w > 25
        
        # Breakout conditions
        breakout_up = price_now > upper_band
        breakout_down = price_now < lower_band
        
        # Entry conditions
        if position == 0:
            # Long: breakout above upper band with volume + trend
            if breakout_up and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: breakout below lower band with volume + trend
            elif breakout_down and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price touches or goes below lower band (reversal signal)
            if price_now <= lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches or goes above upper band (reversal signal)
            if price_now >= upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_1dVolume_1wADX"
timeframe = "12h"
leverage = 1.0
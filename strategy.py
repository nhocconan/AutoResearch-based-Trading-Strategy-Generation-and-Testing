#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1d ADX Trend + Volume Spike + ATR Trailing Stop
Hypothesis: Donchian channel breakouts capture strong momentum. 1d ADX > 25 filters for trending regime (works in bull/bear via direction). Volume spike (>2x 20-period MA) confirms breakout strength. ATR trailing stop (2.5x) manages risk. Designed for 12h timeframe targeting 50-150 total trades over 4 years. Works in both bull and bear markets via ADX regime filter and volume confirmation.
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
    
    # Get 1d data for ADX trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need at least 30 days for ADX
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength filter
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high_1d.diff()
    dm_minus = low_1d.diff().abs() * -1  # inverse for downward movement
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed TR and DM
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = dm_plus.ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = dm_minus.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    
    # ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = dx.ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for trailing stop (12h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Donchian channels for 12h (using previous 20 periods)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start index: need enough for ADX, volume MA, ATR, and Donchian
    start_idx = max(30, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        adx_val = adx_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_val > 25
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Donchian levels
            # Long: price breaks above Donchian high with volume confirmation in trending market
            long_breakout = (curr_close > donchian_high_val) and volume_confirm and trending
            # Short: price breaks below Donchian low with volume confirmation in trending market
            short_breakout = (curr_close < donchian_low_val) and volume_confirm and trending
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: 2.5 * ATR below highest since entry
            trailing_stop = highest_since_entry - 2.5 * atr_val
            # Exit conditions: price closes below Donchian low OR trailing stop hit OR ADX < 20 (trend weakening)
            if curr_close < donchian_low_val or curr_close < trailing_stop or adx_val < 20:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: 2.5 * ATR above lowest since entry
            trailing_stop = lowest_since_entry + 2.5 * atr_val
            # Exit conditions: price closes above Donchian high OR trailing stop hit OR ADX < 20 (trend weakening)
            if curr_close > donchian_high_val or curr_close > trailing_stop or adx_val < 20:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dADX_Trend_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0
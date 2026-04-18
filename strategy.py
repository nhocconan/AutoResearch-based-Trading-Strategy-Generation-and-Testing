#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and daily ADX trend filter.
# Donchian breakout (20-period) provides directional bias; volume > 1.5x 20-period average confirms conviction.
# Daily ADX > 25 ensures we only trade in trending markets, avoiding choppy conditions.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Position size fixed at 0.20 to manage risk. Target: 15-35 trades/year (60-140 total over 4 years).
name = "1h_Donchian20_Volume_ADXTrend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4-period Donchian channels (20-period high/low)
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX (14-period) on 1d data
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    dm_plus = high_1d.diff()
    dm_minus = low_1d.diff().abs()
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    
    # Smoothed DM and TR
    dm_plus_smooth = dm_plus.rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = dm_minus.rolling(window=14, min_periods=14).mean()
    atr_smooth = atr.rolling(window=14, min_periods=14).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_smooth
    di_minus = 100 * dm_minus_smooth / atr_smooth
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Check session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        bull_breakout = close[i] > donchian_high_aligned[i]
        bear_breakout = close[i] < donchian_low_aligned[i]
        
        if position == 0:
            # Long: bullish breakout + volume confirmation + ADX > 25
            if bull_breakout and volume_confirm[i] and adx_aligned[i] > 25:
                signals[i] = 0.20
                position = 1
            # Short: bearish breakout + volume confirmation + ADX > 25
            elif bear_breakout and volume_confirm[i] and adx_aligned[i] > 25:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low OR ADX drops below 20
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR ADX drops below 20
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals
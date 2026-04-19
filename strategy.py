#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly Donchian Breakout with Volume Confirmation and ADX Trend Filter
# Uses weekly Donchian channels (20-week) as major support/resistance levels
# Enters on breakouts with volume confirmation and ADX > 25 to ensure trending conditions
# Exits when price crosses back below/above the opposite Donchian band or ADX weakens
# Works in bull markets (breakouts above upper band) and bear (breakdowns below lower band)
# Target: 15-25 trades/year to avoid fee drag
name = "1d_WeeklyDonchian_Breakout_Volume_ADX_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for multi-timeframe analysis (ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    upper_donchian = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    upper_donchian_aligned = align_htf_to_ltf(prices, df_weekly, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_weekly, lower_donchian)
    
    # Weekly ADX for trend strength (14-period)
    # Calculate True Range
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - np.roll(close_weekly, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_weekly[0] - low_weekly[0]
    # Directional Movement
    dm_plus = np.where((high_weekly - np.roll(high_weekly, 1)) > (np.roll(low_weekly, 1) - low_weekly), 
                       np.maximum(high_weekly - np.roll(high_weekly, 1), 0), 0)
    dm_minus = np.where((np.roll(low_weekly, 1) - low_weekly) > (high_weekly - np.roll(high_weekly, 1)), 
                        np.maximum(np.roll(low_weekly, 1) - low_weekly, 0), 0)
    # Smooth TR, DM+, DM-
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_weekly, adx)
    
    # Daily volume filter: current volume > 1.5x average volume (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or \
           np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.5x average volume
        volume_filter = vol > 1.5 * vol_ma_val
        
        # ADX filter: trending market
        adx_filter = adx_val > 25
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume + ADX
            if price > upper_donchian_aligned[i] and volume_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + volume + ADX
            elif price < lower_donchian_aligned[i] and volume_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below lower Donchian or ADX weakens
            if price < lower_donchian_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above upper Donchian or ADX weakens
            if price > upper_donchian_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
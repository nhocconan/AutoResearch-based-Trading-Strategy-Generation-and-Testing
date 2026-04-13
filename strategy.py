#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
    # Long: price breaks above Donchian upper band (20-period 4h high) + volume > 1.5x 20-period 1d avg + ADX > 25 (trending)
    # Short: price breaks below Donchian lower band (20-period 4h low) + volume > 1.5x 20-period 1d avg + ADX > 25 (trending)
    # Exit: price returns to Donchian midpoint (mean reversion within the channel)
    # Uses 4h primary timeframe for optimal balance of signal quality and trade frequency
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    # Donchian channels provide strong trend-following structure that works in bull and bear markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume and ADX (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros(len(df_1d))
    
    # Calculate Donchian channels on 4h data (20-period)
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    # Middle band = (upper + lower) / 2
    
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    
    donchian_upper = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series_4h.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on 1d data (14-period)
    # ADX measures trend strength regardless of direction
    
    # Calculate True Range
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:]))
    tr = np.concatenate([[np.nan], tr1])  # align length
    
    # Calculate +DM and -DM
    dm_plus = np.where((high_1d[1:] - np.roll(high_1d, 1)[1:]) > (np.roll(low_1d, 1)[1:] - low_1d[1:]),
                       np.maximum(high_1d[1:] - np.roll(high_1d, 1)[1:], 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1)[1:] - low_1d[1:]) > (high_1d[1:] - np.roll(high_1d, 1)[1:]),
                        np.maximum(np.roll(low_1d, 1)[1:] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate +DI and -DI
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align all indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period 1d average
        curr_vol_1d = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmed = curr_vol_1d > 1.5 * vol_avg_20_aligned[i]
        
        # ADX filter: trending market (ADX > 25)
        is_trending = adx_aligned[i] > 25
        
        # Breakout conditions
        breakout_long = (close[i] > donchian_upper_aligned[i] and 
                        volume_confirmed and 
                        is_trending)
        breakout_short = (close[i] < donchian_lower_aligned[i] and 
                         volume_confirmed and 
                         is_trending)
        
        # Exit conditions: return to Donchian midpoint
        exit_long = position == 1 and close[i] <= donchian_middle_aligned[i]
        exit_short = position == -1 and close[i] >= donchian_middle_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0
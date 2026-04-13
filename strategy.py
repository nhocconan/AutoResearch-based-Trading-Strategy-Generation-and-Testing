#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX trend filter and volume confirmation
    # Long: Bull Power > 0 (close > EMA13) + ADX > 25 (trending) + volume > 1.5x 20-period average
    # Short: Bear Power < 0 (close < EMA13) + ADX > 25 (trending) + volume > 1.5x 20-period average
    # Exit: Elder Power crosses zero (mean reversion to EMA13)
    # Uses 6h primary timeframe for lower trade frequency vs 4h, suitable for 6h timeframe constraints
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    # Elder Ray measures bull/bear power relative to EMA, effective in both bull and bear markets when combined with trend filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for volume and ADX (MTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros(len(df_1d))
    
    # Calculate EMA13 on 6h data for Elder Ray
    close_series_6h = pd.Series(close_6h)
    ema13 = close_series_6h.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_6h - ema13
    bear_power = low_6h - ema13
    
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
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_6h, ema13)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(13, n):  # start from 13 to have enough data for EMA calculations
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period 1d average
        curr_vol_1d = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_confirmed = curr_vol_1d > 1.5 * vol_avg_20_aligned[i]
        
        # ADX filter: trending market (ADX > 25)
        is_trending = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = (bull_power_aligned[i] > 0 and 
                     volume_confirmed and 
                     is_trending)
        enter_short = (bear_power_aligned[i] < 0 and 
                      volume_confirmed and 
                      is_trending)
        
        # Exit conditions: Elder Power crosses zero (mean reversion to EMA13)
        exit_long = position == 1 and bull_power_aligned[i] <= 0
        exit_short = position == -1 and bear_power_aligned[i] >= 0
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
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

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0
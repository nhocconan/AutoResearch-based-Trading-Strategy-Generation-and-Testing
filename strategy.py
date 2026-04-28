#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation.
# Enter long when price breaks above 20-period Donchian high, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Enter short when price breaks below 20-period Donchian low, 1d ADX > 25 (trending), and volume > 1.5x 20-bar average.
# Exit when price returns to the 20-period Donchian midpoint (mean of upper and lower bands).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Donchian channels provide clear breakout levels; 1d ADX ensures we only trade in trending regimes (works in bull/bear);
# volume confirmation filters weak breakouts. Designed for lower trade frequency on 12h timeframe.

name = "12h_Donchian20_Breakout_1dADX_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    tr_sum = np.zeros_like(tr)
    dm_plus_sum = np.zeros_like(dm_plus)
    dm_minus_sum = np.zeros_like(dm_minus)
    
    # Initial sum (first 14 periods)
    for i in range(1, tr_period + 1):
        if not np.nanmax([tr[i], dm_plus[i], dm_minus[i]]) != np.nanmax([tr[i], dm_plus[i], dm_minus[i]]):
            pass
        if i < len(tr):
            tr_sum[i] = tr_sum[i-1] + tr[i] if i > 1 else tr[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] + dm_plus[i] if i > 1 else dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] + dm_minus[i] if i > 1 else dm_minus[i]
    
    # Wilder's smoothing
    for i in range(tr_period + 1, len(tr)):
        tr_sum[i] = tr_sum[i-1] - (tr_sum[i-1] / tr_period) + tr[i]
        dm_plus_sum[i] = dm_plus_sum[i-1] - (dm_plus_sum[i-1] / tr_period) + dm_plus[i]
        dm_minus_sum[i] = dm_minus_sum[i-1] - (dm_minus_sum[i-1] / tr_period) + dm_minus[i]
    
    # Avoid division by zero
    tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
    di_plus = 100 * dm_plus_sum / tr_sum_safe
    di_minus = 100 * dm_minus_sum / tr_sum_safe
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1e-10, (di_plus + di_minus))
    
    # ADX: smoothed DX
    adx = np.full_like(dx, np.nan)
    for i in range(tr_period, len(dx)):
        if i == tr_period:
            adx[i] = np.nanmean(dx[1:i+1])
        else:
            adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d ADX trend: > 25 indicates trending market
        adx_trending = adx_aligned[i] > 25
        
        # Price action
        price = close[i]
        
        # Breakout conditions
        breakout_long = price > donchian_high[i]
        breakout_short = price < donchian_low[i]
        
        # Exit conditions: return to midpoint
        exit_long = price < donchian_mid[i]
        exit_short = price > donchian_mid[i]
        
        # Handle entries and exits
        if breakout_long and adx_trending and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_short and adx_trending and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
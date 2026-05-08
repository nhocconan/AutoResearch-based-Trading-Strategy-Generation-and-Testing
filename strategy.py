#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with daily volume confirmation and ADX trend filter.
# Long when price breaks above weekly Donchian upper band AND volume > 1.5x daily average AND ADX > 25 (trending).
# Short when price breaks below weekly Donchian lower band AND volume > 1.5x daily average AND ADX > 25.
# Exit when price crosses back inside the weekly Donchian channel (between upper and lower bands).
# Uses 1d timeframe as specified, with weekly Donchian for higher timeframe context.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag and improve generalization.

name = "1d_WeeklyDonchian_Volume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channel calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    donchian_period = 20
    
    # Upper band: highest high over last 20 weekly periods
    upper_w = pd.Series(high_w).rolling(window=donchian_period, min_periods=donchian_period).max().values
    # Lower band: lowest low over last 20 weekly periods
    lower_w = pd.Series(low_w).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align weekly Donchian levels to daily timeframe
    upper_w_aligned = align_htf_to_ltf(prices, df_w, upper_w)
    lower_w_aligned = align_htf_to_ltf(prices, df_w, lower_w)
    
    # Daily volume filter: current volume > 1.5x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # ADX trend filter (14-period) on daily data
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high[0] - low[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    # Set first DM to 0 (no previous period)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    adx_period = 14
    atr = np.zeros(n)
    dm_plus_smooth = np.zeros(n)
    dm_minus_smooth = np.zeros(n)
    
    # Initial values (simple average of first 'adx_period' periods)
    atr[adx_period-1] = np.mean(tr[:adx_period])
    dm_plus_smooth[adx_period-1] = np.mean(dm_plus[:adx_period])
    dm_minus_smooth[adx_period-1] = np.mean(dm_minus[:adx_period])
    
    # Wilder's smoothing for subsequent values
    for i in range(adx_period, n):
        atr[i] = (atr[i-1] * (adx_period - 1) + tr[i]) / adx_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (adx_period - 1) + dm_plus[i]) / adx_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (adx_period - 1) + dm_minus[i]) / adx_period
    
    # Calculate Directional Indicators
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    dx = np.zeros(n)
    
    # Avoid division by zero
    valid_mask = atr != 0
    di_plus[valid_mask] = (dm_plus_smooth[valid_mask] / atr[valid_mask]) * 100
    di_minus[valid_mask] = (dm_minus_smooth[valid_mask] / atr[valid_mask]) * 100
    
    # Calculate DX and ADX
    dx_denominator = di_plus + di_minus
    valid_dx_mask = dx_denominator != 0
    dx[valid_dx_mask] = (np.abs(di_plus[valid_dx_mask] - di_minus[valid_dx_mask]) / dx_denominator[valid_dx_mask]) * 100
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros(n)
    # Initial ADX value (simple average of first 'adx_period' DX values)
    if len(dx) >= 2*adx_period-1:
        adx[2*adx_period-2] = np.mean(dx[adx_period-1:2*adx_period-1])
        # Wilder's smoothing for subsequent ADX values
        for i in range(2*adx_period-1, n):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Trend filter: ADX > 25
    trend_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(2*adx_period-1, donchian_period, 20)  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_w_aligned[i]) or np.isnan(lower_w_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly Donchian upper band, volume filter, trending market
            long_cond = (close[i] > upper_w_aligned[i]) and volume_filter[i] and trend_filter[i]
            # Short conditions: price breaks below weekly Donchian lower band, volume filter, trending market
            short_cond = (close[i] < lower_w_aligned[i]) and volume_filter[i] and trend_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below weekly Donchian lower band
            if close[i] < lower_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above weekly Donchian upper band
            if close[i] > upper_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
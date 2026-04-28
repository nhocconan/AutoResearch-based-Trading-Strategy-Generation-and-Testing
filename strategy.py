#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Donchian channel breakout with 1d volume confirmation and ADX trend filter.
# Enter long when price breaks above weekly Donchian upper with volume spike and ADX > 25.
# Enter short when price breaks below weekly Donchian lower with volume spike and ADX > 25.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.
# Weekly structure provides strong support/resistance that works in both bull and bear markets.
# Volume confirmation avoids false breakouts. ADX filter ensures we trade only in trending markets.

name = "4h_WeeklyDonchian_Breakout_1dVolume_ADX25_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    n_1w = len(high_1w)
    donchian_upper = np.full(n_1w, np.nan)
    donchian_lower = np.full(n_1w, np.nan)
    
    for i in range(n_1w):
        if i >= 19:  # min_periods=20
            donchian_upper[i] = np.max(high_1w[i-19:i+1])
            donchian_lower[i] = np.min(low_1w[i-19:i+1])
    
    # Forward fill Donchian levels
    donchian_upper = pd.Series(donchian_upper).ffill().values
    donchian_lower = pd.Series(donchian_lower).ffill().values
    
    # Align weekly Donchian to 4h timeframe with 1-bar delay for confirmation
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Get daily data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily volume spike: >2.0x 20-bar average volume
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20
    
    # Align daily volume spike to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Calculate daily ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original arrays
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+ and DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smoothing(tr, 14)
    dm_plus_smooth = wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wilder_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smoothing(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation and ADX filter
        long_breakout = close[i] > donchian_upper_aligned[i] and volume_spike_aligned[i] and adx_aligned[i] > 25
        short_breakout = close[i] < donchian_lower_aligned[i] and volume_spike_aligned[i] and adx_aligned[i] > 25
        
        # Exit conditions: opposite Donchian level or ADX weakening
        long_exit = close[i] < donchian_lower_aligned[i] or adx_aligned[i] < 20
        short_exit = close[i] > donchian_upper_aligned[i] or adx_aligned[i] < 20
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 1d ADX(25) trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper band AND 1d ADX > 25 (strong trend) AND volume > 1.5 * avg_volume(20) on 4h
# Short when price breaks below 12h Donchian lower band AND 1d ADX > 25 (strong trend) AND volume > 1.5 * avg_volume(20) on 4h
# Exit when price crosses back through the 12h Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.25 to balance return and risk, targeting ~100-150 total trades over 4 years (25-37/year)
# 12h Donchian provides wider structure than 4h, reducing whipsaw in choppy markets
# 1d ADX > 25 filter ensures we only trade during strong trending regimes, reducing false breakouts
# Volume confirmation (1.5x) validates breakout strength while avoiding overtrading

name = "4h_12hDonchian20_1dADX25_VolumeConfirm"
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
    
    # Get 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:  # Need at least 21 completed 12h bars for Donchian(20)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid_12h = (donchian_high_12h + donchian_low_12h) / 2.0
    
    # Align 12h Donchian to 4h timeframe (wait for completed 12h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_12h)
    
    # Get 1d data ONCE before loop for ADX(25) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:  # Need at least 26 completed daily bars for ADX(25)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (25-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ and DM- (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(data[1:period])  # Skip first NaN in tr
        # Wilder's smoothing: previous * (period-1)/period + current/period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (period-1)/period + data[i]/period
        return result
    
    atr_1d = wilders_smoothing(tr, 25)
    dm_plus_smooth = wilders_smoothing(dm_plus, 25)
    dm_minus_smooth = wilders_smoothing(dm_minus, 25)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 25)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 12h Donchian high, 1d ADX > 25 (strong trend), volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and 
                adx_1d_aligned[i] > 25.0 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low, 1d ADX > 25 (strong trend), volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and 
                  adx_1d_aligned[i] > 25.0 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 12h Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 12h Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
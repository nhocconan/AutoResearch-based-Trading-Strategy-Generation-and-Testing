#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1d volume spike and ADX regime filter
# Long when price breaks above 1d Donchian upper band AND 1d volume > 2.0 * avg_volume(20) AND 1d ADX > 25
# Short when price breaks below 1d Donchian lower band AND 1d volume > 2.0 * avg_volume(20) AND 1d ADX > 25
# Exit when price crosses 1d Donchian middle band (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian provides daily structure with proven breakout edge
# Volume spike confirms institutional participation (reduces false breakouts)
# ADX > 25 ensures we only trade in trending markets (works in bull/bear regimes)

name = "12h_1dDonchian20_VolumeSpike_ADX_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Donchian, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2.0
        return upper, lower, middle
    
    donchian_upper_1d, donchian_lower_1d, donchian_middle_1d = donchian_channels(high_1d, low_1d, period=20)
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Calculate 1d ADX for regime filter (trending market detection)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = 0  # First period has no prior close
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed averages using Wilder's smoothing
        def wilders_smoothing(values, period):
            result = np.full_like(values, np.nan, dtype=float)
            if len(values) < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(values[:period]) / period
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, period) / atr
        
        # DX and ADX
        dx = np.full_like(close, np.nan, dtype=float)
        mask = (plus_di + minus_di) > 0
        dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
        
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_filter_1d = adx_1d > 25  # Trending market regime
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(adx_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with volume spike and ADX > 25
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with volume spike and ADX > 25
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d Donchian middle (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d Donchian middle (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
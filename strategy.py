#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with volume confirmation and ADX regime filter
# Long when price breaks above 1d Donchian upper band AND 12h volume > 1.5 * avg_volume(20) AND 1d ADX > 20
# Short when price breaks below 1d Donchian lower band AND 12h volume > 1.5 * avg_volume(20) AND 1d ADX > 20
# Exit when price crosses 1d Donchian middle band (mean reversion to equilibrium)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian provides daily structure with proven breakout edge
# Volume confirmation reduces false breakouts (institutional participation)
# ADX > 20 ensures we only trade in trending enough markets (works in bull/bear regimes)

name = "12h_1dDonchian20_12hVolumeSpike_ADX_v1"
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
    
    # Get 1d data ONCE before loop for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for Donchian and ADX
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian(20) channels based on previous 1d bar
    # Upper band = highest high of last 20 periods, Lower band = lowest low of last 20 periods
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donchian_upper_20 = rolling_max(high_1d, 20)
    donchian_lower_20 = rolling_min(low_1d, 20)
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
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
        def wilders_smooth(values, period):
            result = np.full_like(values, np.nan, dtype=float)
            if len(values) < period:
                return result
            # First value: simple average
            result[period-1] = np.nansum(values[:period]) / period
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
            return result
        
        atr = wilders_smooth(tr, period)
        plus_di = 100 * wilders_smooth(plus_dm, period) / atr
        minus_di = 100 * wilders_smooth(minus_dm, period) / atr
        
        # DX and ADX
        dx = np.full_like(close, np.nan, dtype=float)
        mask = (plus_di + minus_di) > 0
        dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask])
        
        adx = wilders_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, period=14)
    adx_filter_1d = adx_1d > 20  # Trending enough market regime
    
    # Get 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume_12h > (1.5 * avg_volume_20_12h)
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_20)
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter_1d)
    
    # Align 12h volume spike to 12h timeframe (no additional delay needed)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(adx_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band with volume spike and ADX > 20
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_spike_aligned[i] and adx_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band with volume spike and ADX > 20
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_spike_aligned[i] and adx_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d Donchian middle band (mean reversion)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d Donchian middle band (mean reversion)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
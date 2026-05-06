#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian channel breakout with volume confirmation and ADX regime filter
# Long when price breaks above 1d Donchian(20) upper band AND volume > 1.5 * avg_volume(20) AND ADX(14) > 25
# Short when price breaks below 1d Donchian(20) lower band AND volume > 1.5 * avg_volume(20) AND ADX(14) > 25
# Exit when price crosses 1d Donchian(20) midline (mean reversion in ranging markets)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian provides strong daily structure with clear breakout levels
# Volume confirmation filters weak breakouts
# ADX filter ensures we only trade in trending markets (avoids whipsaws in ranges)
# Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend)

name = "12h_1dDonchian20_Breakout_Volume_ADX_v1"
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
    if len(df_1d) < 30:  # Need sufficient data for Donchian(20) and ADX(14)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().values
    donchian_middle_1d = (donchian_upper_1d + donchian_lower_1d) / 2.0
    
    # Calculate 1d ADX(14) for trend strength filter
    # ADX calculation requires +DI and -DI
    # +DI = 100 * EWMAS((+DM) / TR, 14)
    # -DI = 100 * EWMAS((-DM) / TR, 14)
    # ADX = 100 * EWMAS(|+DI - -DI| / (+DI + -DI), 14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_series = pd.Series(tr)
    atr_1d = tr_series.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    dm_plus_series = pd.Series(dm_plus)
    dm_minus_series = pd.Series(dm_minus)
    
    di_plus = 100 * dm_plus_series.ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    di_minus = 100 * dm_minus_series.ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # Avoid division by zero
    di_sum = di_plus + di_minus
    di_sum = np.where(di_sum == 0, 1e-10, di_sum)
    dx = 100 * np.abs(di_plus - di_minus) / di_sum
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper band with volume confirmation and ADX > 25
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                volume_confirm[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower band with volume confirmation and ADX > 25
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  volume_confirm[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d Donchian middle band (mean reversion signal)
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d Donchian middle band (mean reversion signal)
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
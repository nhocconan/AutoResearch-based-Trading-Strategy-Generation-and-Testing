#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian(20) breakout with 1d volume spike and 12h ADX trend filter
# Long when price breaks above 1w Donchian upper channel AND 12h ADX > 25 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1w Donchian lower channel AND 12h ADX > 25 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back through 1w Donchian midpoint
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian breakouts capture strong momentum moves
# Volume confirmation validates breakout strength while limiting false signals
# ADX > 25 ensures we only trade in trending markets, reducing whipsaw in ranging conditions
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "12h_1wDonchian20_12hADX25_VolumeSpike"
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
    
    # Get 1w data ONCE before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian(20) channels
    highest_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align 1w Donchian levels to 12h timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume(20) for confirmation threshold
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_threshold_1d = 1.5 * avg_volume_20_1d
    
    # Align 1d volume threshold to 12h timeframe
    volume_threshold_aligned = align_htf_to_ltf(prices, df_1d, volume_threshold_1d)
    
    # Calculate 12h ADX(14) trend filter
    # ADX calculation: +DI, -DI, then DX, then smoothed ADX
    # +DI = 100 * smoothed(+DM) / ATR
    # -DI = 100 * smoothed(-DM) / ATR
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    # ADX = smoothed(DX)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    
    # +DM and -DM
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    plus_di_14 = 100 * wilders_smoothing(plus_dm, 14) / atr_14
    minus_di_14 = 100 * wilders_smoothing(minus_dm, 14) / atr_14
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = wilders_smoothing(dx_14, 14)
    
    # Handle division by zero cases
    plus_di_14 = np.where(atr_14 == 0, 0, plus_di_14)
    minus_di_14 = np.where(atr_14 == 0, 0, minus_di_14)
    dx_14 = np.where((plus_di_14 + minus_di_14) == 0, 0, dx_14)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_threshold_aligned[i]) or 
            np.isnan(adx_14[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper AND ADX > 25 AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                adx_14[i] > 25 and 
                volume[i] > volume_threshold_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower AND ADX > 25 AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  adx_14[i] > 25 and 
                  volume[i] > volume_threshold_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1w Donchian midpoint
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1w Donchian midpoint
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
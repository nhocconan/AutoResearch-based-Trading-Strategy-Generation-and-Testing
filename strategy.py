#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly ADX trend filter and volume confirmation.
# Long when price breaks above 20-day high AND weekly ADX > 25 (strong trend) AND volume > 1.5x 20-day average volume.
# Short when price breaks below 20-day low AND weekly ADX > 25 AND volume > 1.5x 20-day average volume.
# Uses discrete position size 0.25. Donchian provides clear breakout levels, ADX filters for trending markets only.
# Weekly ADX ensures alignment with higher timeframe trend strength. Target: 30-100 trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day average volume for volume confirmation
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Get 1w data once before loop for weekly ADX filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for weekly ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Weekly ADX (14-period) ===
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+ , DM- (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    tr_smoothed = wilders_smoothing(tr, period_adx)
    dm_plus_smoothed = wilders_smoothing(dm_plus, period_adx)
    dm_minus_smoothed = wilders_smoothing(dm_minus, period_adx)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smoothed / tr_smoothed
    di_minus = 100 * dm_minus_smoothed / tr_smoothed
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, period_adx)
    
    # Weekly ADX > 25 indicates strong trend
    weekly_adx_strong = adx > 25
    
    # Align weekly ADX filter to 1d timeframe
    weekly_adx_strong_aligned = align_htf_to_ltf(prices, df_1w, weekly_adx_strong.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods for Donchian, 14+14=28 for ADX)
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(weekly_adx_strong_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high = period20_high[i]
        donch_low = period20_low[i]
        vol = volume[i]
        avg_vol = avg_volume_20[i]
        adx_strong = weekly_adx_strong_aligned[i] > 0.5
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = vol > 1.5 * avg_vol
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price falls below 20-day low (breakdown)
            if price < donch_low:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price rises above 20-day high (breakout)
            if price > donch_high:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 20-day high AND weekly ADX strong AND volume confirmed
            if price > donch_high and adx_strong and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below 20-day low AND weekly ADX strong AND volume confirmed
            elif price < donch_low and adx_strong and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_WeeklyADX25_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0
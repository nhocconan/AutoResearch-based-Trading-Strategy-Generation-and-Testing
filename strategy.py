#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot R4/S4 breakouts with 1d ADX trend filter and volume confirmation.
# Weekly Camarilla R4/S4 represent strong weekly support/resistance levels for breakout/continuation trades.
# Breakout at weekly Camarilla R4 (long) or S4 (short) with volume spike (>1.8x 20-bar average) for confirmation.
# 1d ADX > 25 as trend filter to ensure trades align with strong daily momentum, avoiding choppy markets.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h (within proven winning range).

name = "6h_Camarilla_R4S4_1dADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels (R4, S4)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:  # Need at least 5 weekly bars for reasonable calculation
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R4, S4)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    rng_1w = high_1w - low_1w
    camarilla_r4_1w = close_1w + rng_1w * 1.1 / 2  # R4 level
    camarilla_s4_1w = close_1w - rng_1w * 1.1 / 2  # S4 level
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:  # Need at least 14 days for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(values[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h volume spike: >1.8x 20-bar average volume (stricter to reduce trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Weekly Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r4_aligned[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s4_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend weakening
        long_exit = close[i] < camarilla_s4_aligned[i] or adx_1d_aligned[i] < 20
        short_exit = close[i] > camarilla_r4_aligned[i] or adx_1d_aligned[i] < 20
        
        # Handle entries and exits
        if long_breakout and strong_trend and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and strong_trend and position >= 0:
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
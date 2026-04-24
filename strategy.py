#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 12h ADX(14) > 25 for trending regime (avoid range-bound false breakouts).
- Volume: Current 4h volume > 1.8 * 20-period volume MA to ensure strong participation.
- Entry: Long when price breaks above Donchian upper(20) AND 12h ADX > 25 AND volume spike.
         Short when price breaks below Donchian lower(20) AND 12h ADX > 25 AND volume spike.
- Exit: Opposite Donchian level (lower for long, upper for short) or ADX < 20 (range regime).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide objective breakout levels. ADX ensures we only trade in strong trends,
reducing whipsaws in ranging markets. Volume confirmation avoids low-liquidity breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels for 4h (20-period)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # Using rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 12h ADX(14)
    df_12h_high = df_12h['high'].values
    df_12h_low = df_12h['low'].values
    df_12h_close = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(df_12h_high - df_12h_low)
    tr2 = np.abs(df_12h_high - np.roll(df_12h_close, 1))
    tr3 = np.abs(df_12h_low - np.roll(df_12h_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus = np.where((df_12h_high - np.roll(df_12h_high, 1)) > (np.roll(df_12h_low, 1) - df_12h_low),
                       np.maximum(df_12h_high - np.roll(df_12h_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_12h_low, 1) - df_12h_low) > (df_12h_high - np.roll(df_12h_high, 1)),
                        np.maximum(np.roll(df_12h_low, 1) - df_12h_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero when both DI are zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 20-period volume MA on 12h
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need enough bars for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and strong trend (ADX > 25)
            if volume_spike[i] and adx_val > 25:
                # Bullish: price breaks above Donchian upper
                if curr_high > donchian_upper[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian lower
                elif curr_low < donchian_lower[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower OR ADX < 20 (range regime)
            if curr_low < donchian_lower[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper OR ADX < 20 (range regime)
            if curr_high > donchian_upper[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hADX25_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
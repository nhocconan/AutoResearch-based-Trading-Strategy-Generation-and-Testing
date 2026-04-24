#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w ADX trend filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1w ADX(14) for trend strength (ADX > 25) and 1d EMA50 for trend direction.
- Volume: Current 12h volume > 2.0 * 20-period volume MA to avoid low-volume breakouts.
- Entry: Long when price breaks above Donchian upper band AND ADX > 25 AND close > EMA50.
         Short when price breaks below Donchian lower band AND ADX > 25 AND close < EMA50.
- Exit: Opposite Donchian band (lower for long, upper for short) or ADX < 20 (trend weak).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Donchian channels provide clear breakout levels in trending markets. Combined with ADX filter
to avoid ranging markets and EMA50 for direction, this captures strong trends while
minimizing false signals in choppy conditions. Works in both bull and bear markets by
only taking trades when trend is strong (ADX > 25).
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
    
    # Calculate Donchian channels for 12h (based on previous 20 bars)
    # Upper band = max(high of last 20 periods)
    # Lower band = min(low of last 20 periods)
    # Using rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(df_1w_high - df_1w_low)
    tr2 = np.abs(df_1w_high - np.roll(df_1w_close, 1))
    tr3 = np.abs(df_1w_low - np.roll(df_1w_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    
    # Directional Movement
    dm_plus = np.where((df_1w_high - np.roll(df_1w_high, 1)) > (np.roll(df_1w_low, 1) - df_1w_low),
                       np.maximum(df_1w_high - np.roll(df_1w_high, 1), 0), 0)
    dm_minus = np.where((np.roll(df_1w_low, 1) - df_1w_low) > (df_1w_high - np.roll(df_1w_high, 1)),
                        np.maximum(np.roll(df_1w_low, 1) - df_1w_low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_period = 14
    atr = WilderSmoothing(tr, atr_period)
    dm_plus_smooth = WilderSmoothing(dm_plus, atr_period)
    dm_minus_smooth = WilderSmoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = WilderSmoothing(dx, atr_period)
    adx[:atr_period*2-1] = np.nan  # Not enough data for ADX
    
    # Get 1d data for EMA50 trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need enough bars for Donchian and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        adx_val = adx_aligned[i]
        ema_val = ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and strong trend
            if volume_spike[i] and adx_val > 25:
                # Bullish: price breaks above upper band AND close > EMA50
                if curr_high > donchian_upper[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below lower band AND close < EMA50
                elif curr_low < donchian_lower[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower band OR trend weakens (ADX < 20)
            if curr_low < donchian_lower[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper band OR trend weakens (ADX < 20)
            if curr_high > donchian_upper[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wADX25_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
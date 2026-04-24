#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian channel breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend direction (EMA50) and 1d for volume average and ATR regime.
- Donchian Channel: identifies breakouts from 20-period high/low on 12h timeframe.
- Entry: Long when price breaks above Donchian upper band AND 1w EMA50 is rising AND volume > 1.5 * 20-period average volume (1d).
         Short when price breaks below Donchian lower band AND 1w EMA50 is falling AND volume > 1.5 * 20-period average volume (1d).
- Exit: Opposite Donchian breakout (price crosses back below upper band for longs, above lower band for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian breakouts capture strong momentum moves after consolidation.
- 1w EMA50 filter ensures alignment with weekly trend to avoid counter-trend whipsaws.
- Volume confirmation ensures breakout legitimacy.
- Works in both bull and bear markets by following the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(series, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr_values

def donchian_channel(high, low, period):
    """Calculate Donchian Channel upper and lower bands."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian Channel (20-period)
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = ema(df_1w['close'].values, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_1w_prev = np.roll(ema_50_1w_aligned, 1)
    ema_50_1w_prev[0] = ema_50_1w_aligned[0]
    ema_50_rising = ema_50_1w_aligned > ema_50_1w_prev
    ema_50_falling = ema_50_1w_aligned < ema_50_1w_prev
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d ATR for regime filter (optional volatility filter)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    atr_14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        
        # Exit conditions: price crosses back below upper band for longs, above lower band for shorts
        if position != 0:
            # Exit long: price crosses below upper band
            if position == 1:
                if curr_close < donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above lower band
            elif position == -1:
                if curr_close > donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Donchian breakout signals
            breakout_up = curr_high >= donchian_upper[i] and prev_close < donchian_upper[i-1]
            breakout_down = curr_low <= donchian_lower[i] and prev_close > donchian_lower[i-1]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Trend filter: 1w EMA50 direction
            trend_up = ema_50_rising[i]
            trend_down = ema_50_falling[i]
            
            if breakout_up and volume_confirm and trend_up:
                signals[i] = 0.25
                position = 1
            elif breakout_down and volume_confirm and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
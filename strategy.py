#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d ADX regime filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ADX regime and volume average.
- Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low (measures bull/bear strength relative to trend).
- Entry: Long when Bull Power > 0 AND ADX(14) > 25 (trending market) AND volume > 1.5 * 20-period average volume.
         Short when Bear Power > 0 AND ADX(14) > 25 AND volume > 1.5 * 20-period average volume.
- Exit: When Elder Ray power for current position turns negative OR ADX < 20 (trend weakens).
- Signal size: 0.25 discrete to minimize fee drag.
- Elder Ray captures trend strength via price position relative to EMA.
- ADX filter ensures we only trade in trending markets where Elder Ray is effective.
- Volume confirmation adds validity to the move.
- Works in bull markets (long via Bull Power) and bear markets (short via Bear Power).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period):
    """Calculate Average Directional Index with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = np.where((high_series - high_series.shift(1)) > (low_series.shift(1) - low_series),
                       np.maximum(high_series - high_series.shift(1), 0), 0)
    dm_minus = np.where((low_series.shift(1) - low_series) > (high_series - high_series.shift(1)),
                        np.maximum(low_series.shift(1) - low_series, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean()
    dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean()
    dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # Directional Index
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    
    # ADX
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Handle division by zero or NaN
    adx_values = np.where((di_plus + di_minus) == 0, 0, adx_values)
    adx_values = np.nan_to_num(adx_values, nan=0.0)
    
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX
        return np.zeros(n)
    
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = ema(close, 13)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 30)  # Need 13 for EMA, 20 for volume MA, 30 for ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(ema_13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Elder Ray components
        bull_power = curr_high - ema_13[i]
        bear_power = ema_13[i] - curr_low
        
        # Exit conditions
        if position != 0:
            # Exit long: Bull Power turns negative OR ADX < 20 (weakening trend)
            if position == 1:
                if bull_power <= 0 or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bear Power turns negative OR ADX < 20
            elif position == -1:
                if bear_power <= 0 or adx_1d_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions
        if position == 0:
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # ADX regime filter: ADX > 25 (strong trend)
            strong_trend = adx_1d_aligned[i] > 25
            
            if bull_power > 0 and volume_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
            elif bear_power > 0 and volume_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
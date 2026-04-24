#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h for breakout entries.
- HTF: 1d ADX > 25 for trending regime (avoid whipsaws in ranging markets).
- Volume: Current 6h volume > 1.5 * 20-period 6h volume MA to confirm breakout strength.
- Entry: Long when price breaks above 20-period 6h high AND 1d ADX > 25 AND volume spike.
         Short when price breaks below 20-period 6h low AND 1d ADX > 25 AND volume spike.
- Exit: Opposite Donchian breakout or loss of ADX trend regime (ADX < 20).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Donchian breakouts capture strong momentum moves, while ADX filter ensures we only trade in trending markets,
reducing false breakouts during consolidation. Works in both bull and bear markets by trading the trend direction.
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
    
    # Calculate 6h Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume MA (20-period) for volume confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(df_1d_high[1:] - df_1d_low[1:])
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((df_1d_high[1:] - df_1d_high[:-1]) > (df_1d_low[:-1] - df_1d_low[1:]),
                       np.maximum(df_1d_high[1:] - df_1d_high[:-1], 0), 0)
    dm_minus = np.where((df_1d_low[:-1] - df_1d_low[1:]) > (df_1d_high[1:] - df_1d_high[:-1]),
                        np.maximum(df_1d_low[:-1] - df_1d_low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(data[i]) and not np.isnan(result[i-1]):
                    result[i] = result[i-1] + alpha * (data[i] - result[i-1])
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align HTF indicators to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30)  # Need enough bars for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(vol_ma_6h[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and ADX trend filter
            volume_spike = volume[i] > (1.5 * vol_ma_6h[i])
            if volume_spike and adx_val > 25:
                # Bullish breakout: price above 20-period high
                if curr_high > period20_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below 20-period low
                elif curr_low < period20_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below 20-period low OR ADX < 20 (trend weakening)
            if curr_low < period20_low[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 20-period high OR ADX < 20 (trend weakening)
            if curr_high > period20_high[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dADX_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
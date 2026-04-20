#!/usr/bin/env python3
# 12h_1d_ADX_Trend_With_Volume_Filter
# Hypothesis: Use daily ADX to identify trending markets, enter long when price breaks above 12h Donchian upper band with volume confirmation in uptrends,
# and short when price breaks below 12h Donchian lower band with volume confirmation in downtrends. Exit when price crosses the 12h 20-period EMA.
# This avoids whipsaws in ranging markets (ADX < 25) and focuses on strong trends where breakouts are more reliable.
# Works in bull markets via long breakouts and bear markets via short breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ADX_Trend_With_Volume_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    # === 1d ADX calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    period_adx = 14
    alpha = 1.0 / period_adx
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial values (simple average)
    atr[period_adx-1] = np.mean(tr[:period_adx])
    dm_plus_smooth[period_adx-1] = np.mean(dm_plus[:period_adx])
    dm_minus_smooth[period_adx-1] = np.mean(dm_minus[:period_adx])
    
    # Wilder smoothing
    for i in range(period_adx, len(tr)):
        atr[i] = (atr[i-1] * (period_adx - 1) + tr[i]) / period_adx
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period_adx - 1) + dm_plus[i]) / period_adx
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period_adx - 1) + dm_minus[i]) / period_adx
    
    # Avoid division by zero
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr_safe
    di_minus = 100 * dm_minus_smooth / atr_safe
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = np.zeros_like(dx)
    
    # Initial ADX value
    adx[2*period_adx-1] = np.mean(dx[period_adx:2*period_adx])
    
    # Smooth ADX
    for i in range(2*period_adx, len(dx)):
        adx[i] = (adx[i-1] * (period_adx - 1) + dx[i]) / period_adx
    
    # === 12h Donchian channels ===
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    donchian_period = 20
    upper_band = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 12h 20-period EMA for exit ===
    close_12h = prices['close'].values
    ema_period = 20
    ema_20 = pd.Series(close_12h).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # === 12h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align 1d ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are valid
    start_idx = max(30, donchian_period, ema_period)  # Ensure ADX, Donchian, EMA are ready
    
    for i in range(start_idx, n):
        # Get values
        close_val = close_12h[i]
        upper_val = upper_band[i]
        lower_val = lower_band[i]
        ema_val = ema_20[i]
        vol_ratio_val = vol_ratio[i]
        adx_val = adx_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val) or
            np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending), price breaks above upper Donchian band, volume confirmation
            if (adx_val > 25 and 
                close_val > upper_val and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending), price breaks below lower Donchian band, volume confirmation
            elif (adx_val > 25 and 
                  close_val < lower_val and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below 20-period EMA
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above 20-period EMA
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
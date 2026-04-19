#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) + 1d volume confirmation + 1d ADX trend filter
# Donchian breakout captures trend continuation
# 1d volume > 1.5x 20-period average confirms conviction
# 1d ADX > 25 ensures we only trade in trending markets
# Exit on opposite Donchian band touch
# Designed for 12h timeframe to reduce trade frequency and avoid fee drag
# Target: 15-25 trades/year to stay well within limits
name = "12h_Donchian_ADX_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX (14) - trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        smoothed = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            smoothed[period-1] = np.nansum(values[:period])
            for i in range(period, len(values)):
                smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(tr_14, np.nan, dtype=float)
    di_minus = np.full_like(tr_14, np.nan, dtype=float)
    mask = tr_14 > 0
    di_plus[mask] = 100 * dm_plus_14[mask] / tr_14[mask]
    di_minus[mask] = 100 * dm_minus_14[mask] / tr_14[mask]
    
    # DX and ADX
    dx = np.full_like(tr_14, np.nan, dtype=float)
    dx_mask = (di_plus + di_minus) > 0
    dx[dx_mask] = 100 * np.abs(di_plus[dx_mask] - di_minus[dx_mask]) / (di_plus[dx_mask] + di_minus[dx_mask])
    
    adx = wilders_smoothing(dx, 14)
    adx_14 = adx  # Already smoothed
    
    # Align ADX to 12h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 1d Volume average (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x average
        vol_ma = vol_ma_1d_aligned[i]
        volume_filter = vol_ma > 0 and volume[i] > 1.5 * vol_ma
        
        # Trend filter: ADX > 25
        trend_filter = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume + trend
            if close[i] > donchian_high[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume + trend
            elif close[i] < donchian_low[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches Donchian low
            if close[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian high
            if close[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
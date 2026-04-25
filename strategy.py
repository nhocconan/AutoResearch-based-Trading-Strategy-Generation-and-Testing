#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dRegime_VolumeConfirm_v1
Hypothesis: Trade 6h Donchian(20) breakouts with 1d regime filter (ADX<20 = range, ADX>25 = trend) and volume confirmation (2.0x 24-bar avg). In ranging markets (ADX<20), fade breakouts (sell at upper band, buy at lower band). In trending markets (ADX>25), follow breakouts (buy upper band, sell lower band). Uses discrete sizing (0.25) to limit fee drag and target ~20-40 trades/year. Designed to work in both bull and bear markets by adapting to volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF regime filter and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value: simple average
            result[period-1] = np.nanmean(data[:period])
            # Rest: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=1)
    
    # Calculate Donchian(20) levels from 1d data
    # Upper band = 20-period high, Lower band = 20-period low
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= window - 1:
                result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high_1d, 20)
    donch_low = rolling_min(low_1d, 20)
    
    # Align Donchian levels to 6h timeframe (1-day lagged)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high, additional_delay_bars=1)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 24-bar average volume (48h = 2 days on 6h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX and Donchian
    start_idx = max(34, 20, 24)  # ADX needs ~34 bars (14+14+6), Donchian 20, volume 24
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine regime: ADX < 20 = range, ADX > 25 = trend
        is_range = adx_aligned[i] < 20
        is_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout signals with volume confirmation
            if is_range:
                # In range: fade breakouts (sell at upper band, buy at lower band)
                long_signal = close[i] < donch_low_aligned[i] and volume_spike[i]
                short_signal = close[i] > donch_high_aligned[i] and volume_spike[i]
            else:  # is_trend or neutral (20-25)
                # In trend: follow breakouts (buy upper band, sell lower band)
                long_signal = close[i] > donch_high_aligned[i] and volume_spike[i]
                short_signal = close[i] < donch_low_aligned[i] and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price re-enters the Donchian channel (mean reversion)
            exit_signal = close[i] < donch_high_aligned[i] and close[i] > donch_low_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price re-enters the Donchian channel
            exit_signal = close[i] < donch_high_aligned[i] and close[i] > donch_low_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
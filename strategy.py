#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR calculation and Donchian channel reference.
- Donchian: Upper = 20-period high, Lower = 20-period low on 1d data.
- ATR Regime: Only trade when 1d ATR(14) > 1.5 * ATR(50) to avoid choppy markets.
- Volume Confirmation: Current 12h volume > 1.5 * 20-period average 12h volume.
- Entry: Long when price breaks above 1d Donchian Upper AND volume confirmation AND ATR regime.
         Short when price breaks below 1d Donchian Lower AND volume confirmation AND ATR regime.
- Exit: Opposite Donchian breakout or ATR regime fails (ATR ratio < 1.2).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by capturing breakouts with volatility expansion filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = rolling max(high), lower = rolling min(low)."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_values = tr.rolling(window=period, min_periods=period).mean().values
    return atr_values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR(50)
        return np.zeros(n)
    
    # 1d Donchian(20)
    dc_upper_1d, dc_lower_1d = donchian_channels(
        df_1d['high'].values, 
        df_1d['low'].values, 
        20
    )
    
    # 1d ATR(14) and ATR(50) for regime filter
    atr14_1d = atr(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14
    )
    atr50_1d = atr(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        50
    )
    
    # Align 1d indicators to 12h timeframe
    dc_upper_aligned = align_htf_to_ltf(prices, df_1d, dc_upper_1d)
    dc_lower_aligned = align_htf_to_ltf(prices, df_1d, dc_lower_1d)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(dc_upper_aligned[i]) or np.isnan(dc_lower_aligned[i]) or
            np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # ATR regime: only trade when volatility is expanding (ATR14 > 1.5 * ATR50)
        atr_regime = atr14_aligned[i] > 1.5 * atr50_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_12h[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below Donchian Lower OR ATR regime fails
            if position == 1:
                if curr_close < dc_lower_aligned[i] or not atr_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian Upper OR ATR regime fails
            elif position == -1:
                if curr_close > dc_upper_aligned[i] or not atr_regime:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation and ATR regime
        if position == 0:
            # Long: price breaks above Donchian Upper AND volume confirmation AND ATR regime
            long_condition = (
                curr_close > dc_upper_aligned[i] and
                volume_confirm and
                atr_regime
            )
            
            # Short: price breaks below Donchian Lower AND volume confirmation AND ATR regime
            short_condition = (
                curr_close < dc_lower_aligned[i] and
                volume_confirm and
                atr_regime
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0
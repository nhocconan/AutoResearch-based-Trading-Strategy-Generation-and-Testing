#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR-based regime detection (trending vs ranging).
- Donchian: 20-period high/low channels for breakout signals.
- Entry: Long when price breaks above Donchian(20) high AND ATR(14) > ATR(50) (trending regime) AND volume > 1.5 * 20-period average volume.
         Short when price breaks below Donchian(20) low AND ATR(14) > ATR(50) (trending regime) AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout (price crosses below Donchian(20) low for long exit, above Donchian(20) high for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading in trending regimes (ATR expansion) and using volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper = max(high, period), lower = min(low, period)."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=period, min_periods=period).max().values
    lower = low_series.rolling(window=period, min_periods=period).min().values
    return upper, lower

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

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for regime filter (trending vs ranging)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR50
        return np.zeros(n)
    
    # 1d ATR14 and ATR50 for regime detection
    atr14_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr50_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 50)
    
    # Align ATR components to 4h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high, donchian_low = donchian_channels(high, low, 20)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume MA, 50 for ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: trending market (ATR14 > ATR50 indicates expanding volatility)
        trending_regime = atr14_1d_aligned[i] > atr50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Donchian breakout conditions
        long_breakout = curr_close > donchian_high[i]
        short_breakout = curr_close < donchian_low[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian low
            if position == 1:
                if curr_close < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high
            elif position == -1:
                if curr_close > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with regime and volume filters
        if position == 0:
            # Long: price breaks above Donchian high AND trending regime AND volume confirmation
            long_condition = long_breakout and trending_regime and volume_confirm
            
            # Short: price breaks below Donchian low AND trending regime AND volume confirmation
            short_condition = short_breakout and trending_regime and volume_confirm
            
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

name = "4h_Donchian20_Breakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
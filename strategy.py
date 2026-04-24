#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for pivot direction (bullish if weekly close > weekly open, bearish otherwise) and 1d for ATR volume spike filter.
- Entry: Long when price breaks above Donchian(20) high AND weekly pivot is bullish AND 1d ATR ratio > 1.5.
         Short when price breaks below Donchian(20) low AND weekly pivot is bearish AND 1d ATR ratio > 1.5.
- Exit: Opposite Donchian breakout.
- Signal size: 0.25 discrete to minimize fee drag.
- Weekly pivot provides structural bias to avoid counter-trend trades in ranging markets.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid false breakouts.
- Works in bull markets (buy breakouts in weekly uptrend) and bear markets (sell breakdowns in weekly downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h Donchian(20) channels
    upper, lower = donchian_channels(high, low, 20)
    
    # Calculate 1w pivot direction (bullish if weekly close > weekly open)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_bullish = df_1w['close'].values > df_1w['open'].values  # True if bullish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float), additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below Donchian lower
            if position == 1:
                if curr_close < lower[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper
            elif position == -1:
                if curr_close > upper[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with weekly pivot filter and volume confirmation
        if position == 0:
            # Long: price breaks above Donchian upper AND weekly pivot bullish AND ATR ratio > 1.5
            if curr_close > upper[i] and weekly_bullish_aligned[i] > 0.5 and atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND weekly pivot bearish AND ATR ratio > 1.5
            elif curr_close < lower[i] and weekly_bullish_aligned[i] < 0.5 and atr_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1wPivot_Direction_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
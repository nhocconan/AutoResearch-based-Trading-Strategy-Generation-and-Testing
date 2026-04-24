#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike and ADX trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume spike (ATR ratio) and ADX trend filter.
- Entry: Long when price breaks above Camarilla H3 AND 1d ATR ratio > 1.3 AND 1d ADX > 25.
         Short when price breaks below Camarilla L3 AND 1d ATR ratio > 1.3 AND 1d ADX > 25.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) OR ADX < 20 (trend weak).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide intraday support/resistance based on prior day's range.
- ATR ratio > 1.3 confirms volatility expansion for breakout validity.
- 1d ADX > 25 ensures we trade only in strong trending markets, avoiding chop.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def adx(high, low, close, period):
    """Calculate Average Directional Index."""
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_vals = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    return adx_vals

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (H3, L3, H4, L4)."""
    range_val = high - low
    h3 = close + (range_val * 1.1 / 4)
    l3 = close - (range_val * 1.1 / 4)
    h4 = close + (range_val * 1.1 / 2)
    l4 = close - (range_val * 1.1 / 2)
    return h3, l3, h4, l4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d indicators for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR ratio (current ATR/20-period ATR) for volume spike confirmation
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # 1d ADX for trend strength filter
    adx_vals = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_vals, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    camarilla_high = df_12h['high'].values
    camarilla_low = df_12h['low'].values
    camarilla_close = df_12h['close'].values
    
    h3, l3, _, _ = camarilla_levels(camarilla_high, camarilla_low, camarilla_close)
    h3_aligned = align_htf_to_ltf(prices, df_12h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR ADX < 20 (trend weak)
        if position != 0:
            # Exit long: price breaks below L3 OR ADX falls below 20
            if position == 1:
                if curr_close < l3_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 OR ADX falls below 20
            elif position == -1:
                if curr_close > h3_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above H3 AND ATR ratio > 1.3 AND ADX > 25
            if curr_close > h3_aligned[i] and atr_ratio_aligned[i] > 1.3 and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND ATR ratio > 1.3 AND ADX > 25
            elif curr_close < l3_aligned[i] and atr_ratio_aligned[i] > 1.3 and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_ADXTrend_v1"
timeframe = "12h"
leverage = 1.0
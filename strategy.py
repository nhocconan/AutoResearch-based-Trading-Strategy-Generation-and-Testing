#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d volume spike and 1h ADX trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for Camarilla pivot levels and volume spike filter.
- Additional filter: 1h ADX > 25 to ensure trending market.
- Entry: Long when price breaks above Camarilla H3 AND volume spike AND ADX > 25.
         Short when price breaks below Camarilla L3 AND volume spike AND ADX > 25.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) OR ADX < 20 (range regime).
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels provide institutional support/resistance; volume spike confirms institutional interest;
  ADX filter avoids whipsaws in ranging markets. Works in bull (buy H3 breakouts) and bear (sell L3 breakdowns).
- Estimated trades: ~120 total over 4 years (~30/year) based on strict confluence requirements.
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

def adx(high, low, close, period=14):
    """Calculate Average Directional Index."""
    # True Range
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+
    tr_period = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / (tr_period + 1e-10)
    di_minus = 100 * dm_minus_period / (tr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_values = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return adx_values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla H3/L3 levels
    camarilla_h3 = prev_close + (prev_range * 1.1 / 4)
    camarilla_l3 = prev_close - (prev_range * 1.1 / 4)
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume spike filter (current volume / 20-period average volume)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = df_1d['volume'].values / (vol_ma_20 + 1e-10)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate 1h ADX for trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    adx_values = adx(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1h, adx_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout OR ADX < 20 (range regime)
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR ADX < 20
            if position == 1:
                if curr_close < camarilla_l3_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR ADX < 20
            elif position == -1:
                if curr_close > camarilla_h3_aligned[i] or adx_aligned[i] < 20:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND volume spike AND ADX > 25
            if curr_close > camarilla_h3_aligned[i] and vol_ratio_aligned[i] > 1.5 and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND volume spike AND ADX > 25
            elif curr_close < camarilla_l3_aligned[i] and vol_ratio_aligned[i] > 1.5 and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dVolumeSpike_1hADX_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and chop regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for volume spike and chop regime.
- Entry: Long when price breaks above Camarilla R1 AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime).
         Short when price breaks below Camarilla S1 AND 1d volume > 1.5x 20-period average AND chop > 61.8.
- Exit: Opposite Camarilla breakout OR chop < 38.2 (trend regime) to avoid whipsaw.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels provide intraday support/resistance from prior day.
- Volume spike confirms institutional participation.
- Chop regime filter ensures mean-reversion logic only in ranging markets.
- Works in bull markets (buy R1 breaks in uptrend ranges) and bear markets (sell S1 breaks in downtrend ranges).
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

def true_range(high, low, close):
    """Calculate True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # First period
    return tr

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels for the day."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    return r1, s1

def chop_index(high, low, close, period):
    """Calculate Choppiness Index."""
    atr_sum = pd.Series(true_range(high, low, close)).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (from prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    camarilla_r1, camarilla_s1 = camarilla_levels(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma_20 + 1e-10)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Calculate 1d chop regime filter
    chop = chop_index(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ratio_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Camarilla breakout OR chop < 38.2 (trend regime)
        if position != 0:
            # Exit long: price breaks below S1 OR chop < 38.2 (trend regime)
            if position == 1:
                if curr_close < camarilla_s1_aligned[i] or chop_aligned[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R1 OR chop < 38.2 (trend regime)
            elif position == -1:
                if curr_close > camarilla_r1_aligned[i] or chop_aligned[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and chop regime
        if position == 0:
            # Long: price breaks above R1 AND volume > 1.5x MA AND chop > 61.8 (range regime)
            if curr_close > camarilla_r1_aligned[i] and vol_ratio_aligned[i] > 1.5 and chop_aligned[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 1.5x MA AND chop > 61.8 (range regime)
            elif curr_close < camarilla_s1_aligned[i] and vol_ratio_aligned[i] > 1.5 and chop_aligned[i] > 61.8:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dVolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1-week EMA200 trend filter and 1d volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA200 trend filter, 1d for ATR-based volume spike.
- Entry: Long when price breaks above Camarilla H3 level AND volume spike (ATR ratio > 1.5) AND price > 1w EMA200.
         Short when price breaks below Camarilla L3 level AND volume spike (ATR ratio > 1.5) AND price < 1w EMA200.
- Exit: Opposite Camarilla breakout (L3 for longs, H3 for shorts) OR price crosses 1w EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla levels derived from prior 1d range provide institutional support/resistance.
- Volume spike filter ensures breakouts occur with conviction, reducing false signals.
- 1w EMA200 filter ensures trades align with major trend, working in both bull (buy strength) and bear (sell weakness).
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

def camarilla_levels(high, low, close):
    """
    Calculate Camarilla pivot levels for the day.
    Based on previous day's high, low, close.
    Returns H3, L3, H4, L4 levels.
    """
    range_val = high - low
    H3 = close + (range_val * 1.1 / 4)
    L3 = close - (range_val * 1.1 / 4)
    H4 = close + (range_val * 1.1 / 2)
    L4 = close - (range_val * 1.1 / 2)
    return H3, L3, H4, L4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA200
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:  # Need sufficient data for EMA200
        return np.zeros(n)
    
    ema200_1w = ema(df_1w['close'].values, 200)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_14 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_14 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d data
    # We need to shift by 1 to use prior day's data
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    camarilla_H4 = np.full(n, np.nan)
    camarilla_L4 = np.full(n, np.nan)
    
    for i in range(1, n):
        H3, L3, H4, L4 = camarilla_levels(
            df_1d['high'].values[i-1] if i-1 < len(df_1d) else df_1d['high'].values[-1],
            df_1d['low'].values[i-1] if i-1 < len(df_1d) else df_1d['low'].values[-1],
            df_1d['close'].values[i-1] if i-1 < len(df_1d) else df_1d['close'].values[-1]
        )
        camarilla_H3[i] = H3
        camarilla_L3[i] = L3
        camarilla_H4[i] = H4
        camarilla_L4[i] = L4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3, additional_delay_bars=1)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3, additional_delay_bars=1)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4, additional_delay_bars=1)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA200 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 1w EMA200
            if position == 1:
                if curr_close < camarilla_L3_aligned[i] or curr_close < ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 1w EMA200
            elif position == -1:
                if curr_close > camarilla_H3_aligned[i] or curr_close > ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND volume spike AND bullish 1w trend
            if curr_close > camarilla_H3_aligned[i] and atr_ratio_aligned[i] > 1.5 and curr_close > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND volume spike AND bearish 1w trend
            elif curr_close < camarilla_L3_aligned[i] and atr_ratio_aligned[i] > 1.5 and curr_close < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1wEMA200_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0
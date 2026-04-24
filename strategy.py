#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator breakout with 1w Elder Ray trend filter and 1d ATR volume spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for Elder Ray trend filter (bull/bear power) and 1d for ATR volume confirmation.
- Entry: Long when price breaks above Alligator Lips (5 SMA shifted 3) AND Bull Power > 0 AND ATR ratio > 1.5.
         Short when price breaks below Alligator Teeth (8 SMA shifted 5) AND Bear Power < 0 AND ATR ratio > 1.5.
- Exit: Opposite Alligator breakout OR price crosses Alligator Jaw (13 SMA shifted 8).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid false breakouts.
- Williams Alligator provides dynamic support/resistance with built-in trend alignment.
- Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) from 1w confirms multi-timeframe trend strength.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def sma(values, period):
    """Calculate Simple Moving Average."""
    return pd.Series(values).rolling(window=period, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def alligator(high, low, close):
    """Calculate Williams Alligator: Jaw (13 SMA shifted 8), Teeth (8 SMA shifted 5), Lips (5 SMA shifted 3)."""
    jaw = sma(close, 13)
    teeth = sma(close, 8)
    lips = sma(close, 5)
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Fill shifted values with NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    return jaw, teeth, lips

def elder_ray(high, low, close):
    """Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13."""
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w Elder Ray trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    bull_power, bear_power = elder_ray(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power, additional_delay_bars=1)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 12h Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    jaw, teeth, lips = alligator(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values
    )
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw, additional_delay_bars=1)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth, additional_delay_bars=1)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(atr_ratio_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator breakout OR price crosses Alligator Jaw
        if position != 0:
            # Exit long: price breaks below Alligator Teeth OR price falls below Alligator Jaw
            if position == 1:
                if curr_close < teeth_aligned[i] or curr_close < jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Alligator Lips OR price rises above Alligator Jaw
            elif position == -1:
                if curr_close > lips_aligned[i] or curr_close > jaw_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Alligator Lips AND Bull Power > 0 AND ATR ratio > 1.5
            if curr_close > lips_aligned[i] and bull_power_aligned[i] > 0 and atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Alligator Teeth AND Bear Power < 0 AND ATR ratio > 1.5
            elif curr_close < teeth_aligned[i] and bear_power_aligned[i] < 0 and atr_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1wElderRay_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
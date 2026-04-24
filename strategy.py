#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator with 1d EMA200 trend filter and 1d ATR volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and ATR volume spike filter.
- Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3).
- Entry: Long when Lips > Teeth > Jaw (bullish alignment) AND price > Lips AND ATR ratio > 1.5 AND price > 1d EMA200.
         Short when Jaw > Teeth > Lips (bearish alignment) AND price < Lips AND ATR ratio > 1.5 AND price < 1d EMA200.
- Exit: Opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid false signals.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades in choppy markets.
- Williams Alligator identifies trend beginnings and endings through convergence/divergence of smoothed moving averages.
- Works in bull markets (buy during bullish alignment) and bear markets (sell during bearish alignment).
- Estimated trades: ~80 total over 4 years (~20/year) based on Alligator signal frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Calculate Smoothed Moving Average (SMMA)."""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Williams Alligator components (5, 8, 13 periods)
    # Using 6h data for Alligator
    lips = smma(close, 5)   # Lips: 5-period SMMA
    teeth = smma(close, 8)  # Teeth: 8-period SMMA
    jaw = smma(close, 13)   # Jaw: 13-period SMMA
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d ATR for volume spike filter
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Alligator alignment OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: bearish Alligator alignment OR price falls below 1d EMA200
            if position == 1:
                if (jaw[i] > teeth[i] and teeth[i] > lips[i]) or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish Alligator alignment OR price rises above 1d EMA200
            elif position == -1:
                if (lips[i] > teeth[i] and teeth[i] > jaw[i]) or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Alligator alignment with volatility confirmation and trend filter
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaw
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish alignment: Jaw > Teeth > Lips
            bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
            
            # Long: bullish alignment AND price > Lips AND ATR ratio > 1.5 AND price > 1d EMA200
            if bullish_alignment and curr_close > lips[i] and atr_ratio_aligned[i] > 1.5 and curr_close > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment AND price < Lips AND ATR ratio > 1.5 AND price < 1d EMA200
            elif bearish_alignment and curr_close < lips[i] and atr_ratio_aligned[i] > 1.5 and curr_close < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA200_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
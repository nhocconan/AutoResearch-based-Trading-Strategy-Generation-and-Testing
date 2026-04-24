#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ATR volume spike and 1w EMA34 trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR volume spike confirmation, 1w for EMA34 trend filter.
- Entry: Long when price > Alligator Jaw AND ATR ratio > 1.5 AND price > 1w EMA34.
         Short when price < Alligator Jaw AND ATR ratio > 1.5 AND price < 1w EMA34.
- Exit: Price crosses Alligator Jaw in opposite direction OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3).
  We use Jaw as the main trend indicator.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid false breakouts.
- 1w EMA34 provides higher timeframe trend filter to avoid counter-trend trades.
- Works in bull markets (buy when above Jaw in uptrend) and bear markets (sell when below Jaw in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on volatility expansion frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Calculate Smoothed Moving Average (SMMA)."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean().values
    smma_values = np.full_like(values, np.nan, dtype=float)
    smma_values[period-1] = sma[period-1]
    for i in range(period, len(values)):
        if not np.isnan(smma_values[i-1]):
            smma_values[i] = (smma_values[i-1] * (period-1) + values[i]) / period
        else:
            smma_values[i] = values[i]
    return smma_values

def atr(high, low, close, period):
    """Calculate Average True Range."""
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
    
    # Calculate 12h Williams Alligator Jaw
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Alligator Jaw: 13-period SMMA, shifted 8 bars
    jaw_values = smma(df_12h['close'].values, 13)
    jaw_values_shifted = np.roll(jaw_values, 8)
    jaw_values_shifted[:8] = np.nan  # First 8 values invalid after shift
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_values_shifted, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: price crosses Alligator Jaw OR 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price falls below Jaw OR price falls below 1w EMA34
            if position == 1:
                if curr_close < jaw_12h_aligned[i] or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above Jaw OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > jaw_12h_aligned[i] or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: price relative to Jaw with volatility confirmation and trend filter
        if position == 0:
            # Long: price > Jaw AND ATR ratio > 1.5 AND bullish 1w trend
            if curr_close > jaw_12h_aligned[i] and atr_ratio_aligned[i] > 1.5 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw AND ATR ratio > 1.5 AND bearish 1w trend
            elif curr_close < jaw_12h_aligned[i] and atr_ratio_aligned[i] > 1.5 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0
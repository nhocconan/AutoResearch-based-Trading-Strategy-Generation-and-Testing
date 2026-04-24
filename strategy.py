#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 12h EMA34 trend filter and 1d ATR volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA34 trend filter, 1d for ATR volume spike.
- Entry: Long when Williams %R(14) crosses above -80 from below AND ATR ratio > 1.5 AND price > 12h EMA34.
         Short when Williams %R(14) crosses below -20 from above AND ATR ratio > 1.5 AND price < 12h EMA34.
- Exit: Williams %R crosses below -50 for long OR above -50 for short.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R identifies overbought/oversold conditions; extreme readings (> -80 or < -20) followed by reversal offer high-probability mean reversion spots.
- 12h EMA34 ensures trades align with intermediate-term trend to avoid counter-trend whipsaws.
- 1d ATR ratio > 1.5 confirms volatility expansion to validate reversal strength.
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on Williams %R reversal frequency with strict filters.
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

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Williams %R on 6h (14-period)
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = wr[i]
        prev_wr = wr[i-1] if i > 0 else -50  # Previous Williams %R
        
        # Exit conditions: Williams %R crosses -50 midpoint
        if position != 0:
            # Exit long: Williams %R crosses below -50
            if position == 1:
                if prev_wr > -50 and curr_wr <= -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -50
            elif position == -1:
                if prev_wr < -50 and curr_wr >= -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme reversal with volume confirmation and trend filter
        if position == 0:
            # Long: Williams %R crosses above -80 from below (oversold reversal) AND ATR ratio > 1.5 AND bullish 12h trend
            if prev_wr <= -80 and curr_wr > -80 and atr_ratio_aligned[i] > 1.5 and curr_close > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought reversal) AND ATR ratio > 1.5 AND bearish 12h trend
            elif prev_wr >= -20 and curr_wr < -20 and atr_ratio_aligned[i] > 1.5 and curr_close < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA34_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
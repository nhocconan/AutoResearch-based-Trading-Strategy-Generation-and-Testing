#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d EMA200 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 100-180 total trades over 4 years (25-45/year).
- HTF: 1d for EMA200 trend and volume spike (ATR ratio).
- Entry: Long when Williams %R < -80 (oversold) AND price > 1d EMA200 AND ATR ratio > 1.5.
         Short when Williams %R > -20 (overbought) AND price < 1d EMA200 AND ATR ratio > 1.5.
- Exit: Williams %R crosses above -50 (for long) or below -50 (for short) OR price crosses 1d EMA200 opposite.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R identifies exhaustion points in both bull and bear markets.
- 1d EMA200 ensures trades align with higher timeframe trend.
- ATR ratio > 1.5 confirms volatility expansion to avoid false reversals.
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
- Estimated trades: ~140 total over 4 years (~35/year) based on extreme %R frequency with strict filters.
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
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Williams %R on 4h (14-period)
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: Williams %R crosses -50 OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: Williams %R rises above -50 OR price falls below 1d EMA200
            if position == 1:
                if wr[i] > -50 or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R falls below -50 OR price rises above 1d EMA200
            elif position == -1:
                if wr[i] < -50 or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend filter and volume confirmation
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND bullish 1d trend AND volatility expansion
            if wr[i] < -80 and curr_close > ema200_1d_aligned[i] and atr_ratio_aligned[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND bearish 1d trend AND volatility expansion
            elif wr[i] > -20 and curr_close < ema200_1d_aligned[i] and atr_ratio_aligned[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA200_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
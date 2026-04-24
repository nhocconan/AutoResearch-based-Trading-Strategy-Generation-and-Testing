#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d volume spike and 1d EMA trend filter.
- Primary timeframe: 6h targeting 75-150 total trades over 4 years (19-37/year).
- HTF: 1d for Camarilla levels, ATR volume spike, and EMA trend.
- Entry: Long when price breaks above Camarilla R3 AND 1d ATR ratio > 1.3 AND price > 1d EMA34.
         Short when price breaks below Camarilla S3 AND 1d ATR ratio > 1.3 AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout (R3/S3) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla pivots identify key intraday support/resistance levels from prior 1d session.
- ATR ratio (current ATR/14-period ATR) > 1.3 confirms volatility expansion for breakout validity.
- 1d EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy R3 breakouts in uptrend) and bear markets (sell S3 breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with filters.
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
    """Calculate Camarilla pivot levels for the day."""
    pivot = (high + low + close) / 3
    range_ = high - low
    r3 = pivot + range_ * 1.1 / 4
    s3 = pivot - range_ * 1.1 / 4
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR ratio for volume spike filter (current ATR / 14-period ATR)
    atr_14 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_14 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d Camarilla levels (R3, S3) from prior day's OHLC
    camarilla_r3, camarilla_s3 = camarilla_levels(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla S3 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < camarilla_s3_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla R3 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > camarilla_r3_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla R3 AND ATR ratio > 1.3 AND bullish 1d trend
            if curr_close > camarilla_r3_aligned[i] and atr_ratio_aligned[i] > 1.3 and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND ATR ratio > 1.3 AND bearish 1d trend
            elif curr_close < camarilla_s3_aligned[i] and atr_ratio_aligned[i] > 1.3 and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dATR_VolumeSpike_1dEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0
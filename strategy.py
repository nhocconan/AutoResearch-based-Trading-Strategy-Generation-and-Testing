#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1w EMA50 trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation (ATR spike) and 1w for EMA50 trend filter.
- Entry: Long when price breaks above Camarilla R3 AND 1d ATR ratio > 1.8 AND price > 1w EMA50.
         Short when price breaks below Camarilla S3 AND 1d ATR ratio > 1.8 AND price < 1w EMA50.
- Exit: Opposite Camarilla breakout (R4/S4) OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Camarilla pivot levels provide intraday support/resistance with statistical edge.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms volatility expansion for breakout validity.
- 1w EMA50 provides strong trend filter to avoid counter-trend trades in ranging markets.
- Works in bull markets (buy R3 breakouts in uptrend) and bear markets (sell S3 breakdowns in downtrend).
- Estimated trades: ~80 total over 4 years (~20/year) based on Camarilla breakout frequency with filters.
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

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels (R3, R4, S3, S4)."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r3 = pivot + (range_hl * 1.1 / 4.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 12h
    r3_12h, r4_12h, s3_12h, s4_12h = camarilla_pivot(high, low, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r3_12h[i]) or np.isnan(r4_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks above R4 OR price falls below 1w EMA50
            if position == 1:
                if curr_close > r4_12h[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks below S4 OR price rises above 1w EMA50
            elif position == -1:
                if curr_close < s4_12h[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above R3 AND ATR ratio > 1.8 AND bullish 1w trend
            if curr_close > r3_12h[i] and atr_ratio_aligned[i] > 1.8 and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND ATR ratio > 1.8 AND bearish 1w trend
            elif curr_close < s3_12h[i] and atr_ratio_aligned[i] > 1.8 and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dATR_VolumeSpike_1wEMA50_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0
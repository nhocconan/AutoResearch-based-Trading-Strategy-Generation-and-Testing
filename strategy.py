#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX zero-cross with 12h EMA50 trend filter and volume confirmation using 1d ATR spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter and 1d for ATR volume spike filter.
- Entry: Long when TRIX crosses above zero AND ATR ratio > 2.0 AND price > 12h EMA50.
         Short when TRIX crosses below zero AND ATR ratio > 2.0 AND price < 12h EMA50.
- Exit: Opposite TRIX zero-cross OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- TRIX (12,26,9) is a momentum oscillator that filters noise and identifies trend changes.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false signals.
- 12h EMA50 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy momentum in uptrend) and bear markets (sell momentum in downtrend).
- Estimated trades: ~120 total over 4 years (~30/year) based on momentum shift frequency with strict filters.
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

def trix(close, period):
    """Calculate TRIX indicator."""
    # Triple exponential moving average
    ema1 = pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    # Calculate percentage change
    trix_values = pd.Series(ema3).pct_change() * 100
    return trix_values.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate TRIX (12,26,9) on 4h data
    trix_values = trix(close, 12)
    trix_signal = trix_values  # TRIX line itself
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(trix_signal[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_trix = trix_signal[i]
        prev_trix = trix_signal[i-1] if i > 0 else 0
        
        # Exit conditions: opposite TRIX zero-cross OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: TRIX crosses below zero OR price falls below 12h EMA50
            if position == 1:
                if curr_trix < 0 and prev_trix >= 0 or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: TRIX crosses above zero OR price rises above 12h EMA50
            elif position == -1:
                if curr_trix > 0 and prev_trix <= 0 or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: TRIX zero-cross with volatility confirmation and trend filter
        if position == 0:
            # Long: TRIX crosses above zero AND ATR ratio > 2.0 AND bullish 12h trend
            if curr_trix > 0 and prev_trix <= 0 and atr_ratio_aligned[i] > 2.0 and curr_close > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND ATR ratio > 2.0 AND bearish 12h trend
            elif curr_trix < 0 and prev_trix >= 0 and atr_ratio_aligned[i] > 2.0 and curr_close < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_TRIX_ZeroCross_1dATR_VolumeSpike_12hEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: Daily ATR-based volatility expansion combined with 1-week EMA50 trend filter and Donchian(20) breakout confirmation.
- Primary timeframe: 1d targeting 70-120 total trades over 4 years (17-30/year).
- HTF: 1w for EMA50 trend filter.
- Entry: Long when ATR ratio > 2.2 (volatility spike) AND price breaks above Donchian(20) high AND price > 1w EMA50.
         Short when ATR ratio > 2.2 AND price breaks below Donchian(20) low AND price < 1w EMA50.
- Exit: ATR ratio < 1.2 (volatility contraction) OR opposite Donchian breakout.
- Signal size: 0.25 discrete to minimize fee drag.
- ATR ratio = current ATR(1) / ATR(20); values > 2.2 indicate significant volatility expansion to avoid choppy markets.
- 1w EMA50 provides trend filter to align with higher timeframe momentum.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend) by requiring trend alignment.
- Estimated trades: ~90 total over 4 years (~22/year) based on volatility breakout frequency with strict filters.
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

def donchian_channels(high, low, period):
    """Calculate Donchian Channels."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Calculate 1d ATR for volatility spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Donchian channels on 1d (20-period)
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: volatility contraction OR opposite Donchian breakout
        if position != 0:
            # Exit any position: ATR ratio < 1.2 (volatility contraction) OR opposite Donchian breakout
            if atr_ratio_aligned[i] < 1.2:
                signals[i] = 0.0
                position = 0
                continue
            # Exit long: price breaks below Donchian low
            elif position == 1 and curr_close < donch_lo[i]:
                signals[i] = 0.0
                position = 0
                continue
            # Exit short: price breaks above Donchian high
            elif position == -1 and curr_close > donch_hi[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: volatility expansion + Donchian breakout + trend filter
        if position == 0:
            # Long: ATR ratio > 2.2 AND price breaks above Donchian high AND bullish 1w trend
            if atr_ratio_aligned[i] > 2.2 and curr_close > donch_hi[i] and curr_close > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: ATR ratio > 2.2 AND price breaks below Donchian low AND bearish 1w trend
            elif atr_ratio_aligned[i] > 2.2 and curr_close < donch_lo[i] and curr_close < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_ATR_VolatilityExpansion_DonchianBreakout_1wEMA50_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0
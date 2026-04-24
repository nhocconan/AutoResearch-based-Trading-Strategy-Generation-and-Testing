#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation using 1d ATR spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and ATR volume spike filter.
- Entry: Long when price breaks above Donchian upper band (20-period high) AND ATR ratio > 2.0 AND price > 1d EMA50.
         Short when price breaks below Donchian lower band (20-period low) AND ATR ratio > 2.0 AND price < 1d EMA50.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA50 provides trend filter to avoid counter-trend trades.
- Donchian channels provide objective price structure that works in both trending and ranging markets.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
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
    """Calculate Donchian channels (upper, lower bands)."""
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
    
    # Calculate 1d trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    ema50_1d = ema(df_1d['close'].values, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    upper_20, lower_20 = donchian_channels(df_12h['high'].values, df_12h['low'].values, 20)
    upper_20_aligned = align_htf_to_ltf(prices, df_12h, upper_20, additional_delay_bars=1)
    lower_20_aligned = align_htf_to_ltf(prices, df_12h, lower_20, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 70  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian lower band OR price falls below 1d EMA50
            if position == 1:
                if curr_close < lower_20_aligned[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper band OR price rises above 1d EMA50
            elif position == -1:
                if curr_close > upper_20_aligned[i] or curr_close > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Donchian upper band AND ATR ratio > 2.0 AND bullish 1d trend
            if curr_close > upper_20_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND ATR ratio > 2.0 AND bearish 1d trend
            elif curr_close < lower_20_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_TrendFilter_1dATR_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0
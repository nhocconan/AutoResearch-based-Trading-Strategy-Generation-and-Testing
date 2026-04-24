#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and ATR-based volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter and ATR-based volume spike filter.
- Entry: Long when price breaks above Donchian upper(20) AND ATR(1)/ATR(20) > 1.8 AND price > 12h EMA50.
         Short when price breaks below Donchian lower(20) AND ATR(1)/ATR(20) > 1.8 AND price < 12h EMA50.
- Exit: Opposite Donchian breakout OR price crosses 12h EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide clear structure and breakout signals.
- ATR ratio > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 12h EMA50 provides medium-term trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on volatility breakout frequency with strict filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def atr(high, low, close, period):
    """Calculate Average True Range with proper min_periods."""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    return pd.Series(true_range).ewm(span=period, adjust=False, min_periods=period).mean().values

def donchian_channels(high, low, period):
    """Calculate Donchian channels (upper, lower)."""
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
    
    # Calculate 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_12h = ema(df_12h['close'].values, 50)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Calculate 12h ATR for volume spike filter
    if len(df_12h) < 30:
        return np.zeros(n)
    
    atr_20_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 20)
    atr_current_12h = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 1)
    atr_ratio_12h = atr_current_12h / (atr_20_12h + 1e-10)  # Avoid division by zero
    atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h, additional_delay_bars=1)
    
    # Donchian channels (20-period)
    upper, lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 30, 20)  # Need sufficient data for EMA50, ATR, and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(atr_ratio_12h_aligned[i]) or
            np.isnan(upper[i]) or np.isnan(lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 12h EMA50 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian lower OR price falls below 12h EMA50
            if position == 1:
                if curr_close < lower[i] or curr_close < ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper OR price rises above 12h EMA50
            elif position == -1:
                if curr_close > upper[i] or curr_close > ema50_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Donchian upper AND ATR ratio > 1.8 AND bullish 12h trend
            if curr_close > upper[i] and atr_ratio_12h_aligned[i] > 1.8 and curr_close > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND ATR ratio > 1.8 AND bearish 12h trend
            elif curr_close < lower[i] and atr_ratio_12h_aligned[i] > 1.8 and curr_close < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_TrendFilter_12hATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
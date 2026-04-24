#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR volume spike and 1w EMA50 trend filter.
- Primary timeframe: 6h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR-based volume spike confirmation, 1w for EMA50 trend filter.
- Entry: Long when price breaks above Donchian(20) high AND 1d ATR(7)/ATR(30) > 2.0 AND price > 1w EMA50.
         Short when price breaks below Donchian(20) low AND 1d ATR(7)/ATR(30) > 2.0 AND price < 1w EMA50.
- Exit: Opposite Donchian breakout OR price crosses 1w EMA50 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels identify breakouts with clear structure.
- ATR ratio > 2.0 confirms volatility expansion (institutional participation).
- 1w EMA50 ensures trading with the higher timeframe trend.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
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

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian(20) channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 1d ATR ratio: ATR(7)/ATR(30) for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    atr7_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 7)
    atr30_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 30)
    atr_ratio_1d = atr7_1d / np.where(atr30_1d == 0, 1e-10, atr30_1d)  # Avoid division by zero
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Breakout conditions
    bullish_breakout = close > donchian_high  # Price breaks above Donchian high
    bearish_breakout = close < donchian_low   # Price breaks below Donchian low
    
    # Volume spike condition: ATR ratio > 2.0 indicates volatility expansion
    volume_spike = atr_ratio_1d_aligned > 2.0
    
    # Trend filter: price relative to 1w EMA50
    trend_bullish = close > ema50_1w_aligned
    trend_bearish = close < ema50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_period, 60)  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite breakout OR price crosses 1w EMA50 in opposite direction
        if position != 0:
            # Exit long: bearish breakout OR price falls below 1w EMA50
            if position == 1:
                if bearish_breakout[i] or curr_close < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: bullish breakout OR price rises above 1w EMA50
            elif position == -1:
                if bullish_breakout[i] or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: breakout with volume spike and trend alignment
        if position == 0:
            # Long: bullish breakout AND volume spike AND bullish 1w trend
            if bullish_breakout[i] and volume_spike[i] and trend_bullish[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout AND volume spike AND bearish 1w trend
            elif bearish_breakout[i] and volume_spike[i] and trend_bearish[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATR_VolumeSpike_1wEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0
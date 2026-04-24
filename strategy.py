#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA200 trend filter and volume spike confirmation using 12h ATR ratio.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA200 trend filter and 12h for ATR volume spike.
- Entry: Long when price breaks above Donchian(20) high AND ATR ratio > 1.8 AND price > 1d EMA200.
         Short when price breaks below Donchian(20) low AND ATR ratio > 1.8 AND price < 1d EMA200.
- Exit: Opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA200 provides strong trend filter to avoid counter-trend trades and adapt to bull/bear markets.
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
    
    # Calculate 1d trend filter: EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:
        return np.zeros(n)
    
    ema200_1d = ema(df_1d['close'].values, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d, additional_delay_bars=1)
    
    # Calculate 12h ATR for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 20)
    atr_current = atr(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio, additional_delay_bars=1)
    
    # Donchian channels on 12h (20-period)
    donch_hi, donch_lo = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 210  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donch_hi[i]) or np.isnan(donch_lo[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses 1d EMA200 in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian low OR price falls below 1d EMA200
            if position == 1:
                if curr_close < donch_lo[i] or curr_close < ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR price rises above 1d EMA200
            elif position == -1:
                if curr_close > donch_hi[i] or curr_close > ema200_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Donchian high AND ATR ratio > 1.8 AND bullish 1d trend
            if curr_close > donch_hi[i] and atr_ratio_aligned[i] > 1.8 and curr_close > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND ATR ratio > 1.8 AND bearish 1d trend
            elif curr_close < donch_lo[i] and atr_ratio_aligned[i] > 1.8 and curr_close < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_12hATR_VolumeSpike_1dEMA200_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0
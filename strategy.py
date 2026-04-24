#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR volume spike and 1w EMA34 trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter, 1d for ATR volume spike.
- Entry: Long when price breaks above Camarilla H3 AND ATR ratio > 2.0 AND price > 1w EMA34.
         Short when price breaks below Camarilla L3 AND ATR ratio > 2.0 AND price < 1w EMA34.
- Exit: Opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 1w EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Camarilla levels derived from prior 1d session, ensuring no look-ahead.
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

def camarilla_levels(high, low, close):
    """
    Calculate Camarilla pivot levels for today based on yesterday's OHLC.
    H3 = close + 1.1*(high-low)/2
    L3 = close - 1.1*(high-low)/2
    """
    range_ = high - low
    h3 = close + 1.1 * range_ / 2
    l3 = close - 1.1 * range_ / 2
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1w trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    ema34_1w = ema(df_1w['close'].values, 34)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from prior 1d session (shifted by 1 to avoid look-ahead)
    # We need to shift the 1d data by 1 bar to get prior session's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use prior session's data for today's levels
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but alignment will handle timing
    
    camarilla_h3_1d, camarilla_l3_1d = camarilla_levels(high_1d_shifted, low_1d_shifted, close_1d_shifted)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 1w EMA34
            if position == 1:
                if curr_close < camarilla_l3_aligned[i] or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > camarilla_h3_aligned[i] or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 2.0 AND bullish 1w trend
            if curr_close > camarilla_h3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 2.0 AND bearish 1w trend
            elif curr_close < camarilla_l3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d ATR volume spike and 1d EMA34 trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and ATR volume spike confirmation.
- Entry: Long when price breaks above Camarilla H3 AND ATR ratio > 1.8 AND price > 1d EMA34.
         Short when price breaks below Camarilla L3 AND ATR ratio > 1.8 AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout (H4/L4) OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA34 provides trend filter to avoid counter-trend trades.
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

def camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels."""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    h3 = pivot + (range_ * 1.1 / 4)
    l3 = pivot - (range_ * 1.1 / 4)
    h4 = pivot + (range_ * 1.1 / 2)
    l4 = pivot - (range_ * 1.1 / 2)
    return h3, l3, h4, l4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 6h data
    h3 = np.full(n, np.nan)
    l3 = np.full(n, np.nan)
    h4 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:  # Need at least one bar for pivot calculation
            continue
        h3[i], l3[i], h4[i], l4[i] = camarilla_pivot(high[i-1], low[i-1], close[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout (H4/L4) OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L4 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < l4[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H4 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > h4[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.8 AND bullish 1d trend
            if curr_close > h3[i] and atr_ratio_aligned[i] > 1.8 and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.8 AND bearish 1d trend
            elif curr_close < l3[i] and atr_ratio_aligned[i] > 1.8 and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1dEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR volume spike filter and 12h EMA34 trend filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR volume spike, 12h for EMA34 trend filter.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 2.0 AND price > 12h EMA34.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 2.0 AND price < 12h EMA34.
- Exit: Opposite Camarilla breakout (L3 for long exit, H3 for short exit) OR price crosses 12h EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- ATR ratio (current 1d ATR/20-period 1d ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 12h EMA34 provides trend filter to avoid counter-trend trades.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Uses Camarilla H3/L3 levels which are proven effective reversal/breakout levels from institutional trading.
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
    """Calculate Camarilla pivot levels (H3, L3)."""
    range_val = high - low
    camarilla_h3 = close + range_val * 1.1 / 4
    camarilla_l3 = close - range_val * 1.1 / 4
    return camarilla_h3, camarilla_l3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d OHLC
    camarilla_h3, camarilla_l3 = camarilla_levels(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3, additional_delay_bars=1)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema34_12h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 12h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 12h EMA34
            if position == 1:
                if curr_close < camarilla_l3_aligned[i] or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 12h EMA34
            elif position == -1:
                if curr_close > camarilla_h3_aligned[i] or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 2.0 AND bullish 12h trend
            if curr_close > camarilla_h3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 2.0 AND bearish 12h trend
            elif curr_close < camarilla_l3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_12hEMA34_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
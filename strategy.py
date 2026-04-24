#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla pivot (H3/L3) breakout with 1-week EMA34 trend filter and 1-day ATR volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for EMA34 trend filter, 1d for ATR volume spike.
- Entry: Long when price breaks above Camarilla H3 AND ATR ratio > 2.0 AND price > 1w EMA34.
         Short when price breaks below Camarilla L3 AND ATR ratio > 2.0 AND price < 1w EMA34.
- Exit: Opposite Camarilla breakout (L3 for long exit, H3 for short exit) OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.30 discrete to balance profit potential and fee drag.
- ATR ratio (current ATR/20-period ATR) > 2.0 confirms significant volatility expansion to avoid false breakouts.
- 1w EMA34 provides smooth trend filter to avoid counter-trend trades in choppy markets.
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
    """Calculate Camarilla pivot levels (H3, L3) from previous day's OHLC."""
    # Camarilla levels based on previous day's range
    pivot = (high + low + close) / 3.0
    range_val = high - low
    H3 = pivot + (range_val * 1.1 / 4.0)  # equivalent to pivot + range * 1.1/4
    L3 = pivot - (range_val * 1.1 / 4.0)  # equivalent to pivot - range * 1.1/4
    return H3, L3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate 1d ATR for volume spike filter (using previous day's data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d data (shifted by 1 to avoid look-ahead)
    # We need previous day's OHLC for today's levels
    df_1d_prev = df_1d.copy()
    # Shift OHLC by 1 to get previous day's values
    high_prev = np.roll(df_1d['high'].values, 1)
    low_prev = np.roll(df_1d['low'].values, 1)
    close_prev = np.roll(df_1d['close'].values, 1)
    # First value will be invalid (rolled from last), set to NaN
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_H3, camarilla_L3 = camarilla_pivot(high_prev, low_prev, close_prev)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3, additional_delay_bars=1)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 1w EMA34
            if position == 1:
                if curr_close < camarilla_L3_aligned[i] or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > camarilla_H3_aligned[i] or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 2.0 AND bullish 1w trend
            if curr_close > camarilla_H3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 2.0 AND bearish 1w trend
            elif curr_close < camarilla_L3_aligned[i] and atr_ratio_aligned[i] > 2.0 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.30
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0
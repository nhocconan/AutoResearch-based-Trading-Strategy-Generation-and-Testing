#!/usr/bin/env python3
"""
Hypothesis: Daily Camarilla pivot breakout with volume confirmation and weekly trend filter.
- Primary timeframe: 1d targeting 30-80 total trades over 4 years (7-20/year).
- HTF: 1w EMA34 for trend filter.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 1.5 AND price > 1w EMA34.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 1.5 AND price < 1w EMA34.
- Exit: Opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Camarilla levels provide institutional support/resistance levels that work in both ranging and trending markets.
- Volume confirmation via ATR spike filters out low-momentum false breakouts.
- Weekly EMA34 trend filter ensures we trade with the higher-timeframe momentum.
- Works in bull markets (buy H3 breakouts in uptrend) and bear markets (sell L3 breakdowns in downtrend).
- Estimated trades: ~50 total over 4 years (~12-13/year) based on Camarilla breakout frequency with strict filters.
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
    """Calculate Camarilla pivot levels for intraday trading."""
    pivot = (high + low + close) / 3
    range_val = high - low
    h4 = pivot + (range_val * 1.1 / 2)
    h3 = pivot + (range_val * 1.1 / 4)
    h2 = pivot + (range_val * 1.1 / 6)
    h1 = pivot + (range_val * 1.1 / 12)
    l1 = pivot - (range_val * 1.1 / 12)
    l2 = pivot - (range_val * 1.1 / 6)
    l3 = pivot - (range_val * 1.1 / 4)
    l4 = pivot - (range_val * 1.1 / 2)
    return h3, l3  # We only need H3 and L3 for breakouts

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
    
    # Calculate Camarilla levels on 1d (using previous day's data to avoid look-ahead)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, n):  # Start from 1 to have previous day's data
        h3, l3 = camarilla_levels(high[i-1], low[i-1], close[i-1])
        camarilla_h3[i] = h3
        camarilla_l3[i] = l3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1w EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 1w EMA34
            if position == 1:
                if curr_close < camarilla_l3[i] or curr_close < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 1w EMA34
            elif position == -1:
                if curr_close > camarilla_h3[i] or curr_close > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.5 AND bullish 1w trend
            if curr_close > camarilla_h3[i] and atr_ratio_aligned[i] > 1.5 and curr_close > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.5 AND bearish 1w trend
            elif curr_close < camarilla_l3[i] and atr_ratio_aligned[i] > 1.5 and curr_close < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1wEMA34_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0
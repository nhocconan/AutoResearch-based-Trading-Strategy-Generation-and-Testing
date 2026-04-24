#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d ATR volume spike and 1d EMA34 trend filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and ATR volume confirmation.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 1.8 AND price > 1d EMA34.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 1.8 AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR ratio (current ATR/20-period ATR) > 1.8 confirms significant volatility expansion to avoid false breakouts.
- 1d EMA34 provides trend filter to avoid counter-trend trades.
- Camarilla levels derived from previous 1d session (high, low, close) provide institutional support/resistance.
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

def camarilla_levels(high, low, close):
    """
    Calculate Camarilla pivot levels for intraday trading.
    Based on previous period's high, low, close.
    Returns: H4, H3, H2, H1, L1, L2, L3, L4
    We use H3 and L3 for breakout trading.
    """
    range_val = high - low
    h3 = close + range_val * 1.1 / 4
    l3 = close - range_val * 1.1 / 4
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    atr_20_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current_1d = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio_1d = atr_current_1d / (atr_20_1d + 1e-10)  # Avoid division by zero
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d, additional_delay_bars=1)
    
    # Calculate Camarilla levels from 1d data (previous day's HLC)
    # We need to shift the 1d data by 1 to avoid look-ahead
    h3_1d, l3_1d = camarilla_levels(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    # Align Camarilla levels to 12h timeframe (no additional delay needed as they're based on completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr_ratio_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below L3 level OR price falls below 1d EMA34
            if position == 1:
                if curr_close < l3_1d_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above H3 level OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > h3_1d_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above H3 level AND ATR ratio > 1.8 AND bullish 1d trend
            if curr_close > h3_1d_aligned[i] and atr_ratio_1d_aligned[i] > 1.8 and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 level AND ATR ratio > 1.8 AND bearish 1d trend
            elif curr_close < l3_1d_aligned[i] and atr_ratio_1d_aligned[i] > 1.8 and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1dEMA34_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0
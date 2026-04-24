#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation using 1d ATR spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter and ATR volume spike.
- Entry: Long when price breaks above Camarilla H3 level AND ATR ratio > 1.8 AND price > 1d EMA34.
         Short when price breaks below Camarilla L3 level AND ATR ratio > 1.8 AND price < 1d EMA34.
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

def camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels for intraday trading."""
    # Camarilla formula based on previous day's range
    pivot = (high + low + close) / 3
    range_val = high - low
    
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    h4 = pivot + (range_val * 1.1 / 2)
    l4 = pivot - (range_val * 1.1 / 2)
    
    return h3, l3, h4, l4

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
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    # Calculate Camarilla levels on 4h using previous 1d OHLC
    # We need to shift the 1d data by 1 to avoid look-ahead (use previous day's levels)
    h3, l3, h4, l4 = camarilla_levels(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3, additional_delay_bars=1)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3, additional_delay_bars=1)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4, additional_delay_bars=1)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Camarilla breakout (H4/L4) OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L4 OR price falls below 1d EMA34
            if position == 1:
                if curr_close < l4_aligned[i] or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H4 OR price rises above 1d EMA34
            elif position == -1:
                if curr_close > h4_aligned[i] or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 level AND ATR ratio > 1.8 AND bullish 1d trend
            if curr_close > h3_aligned[i] and atr_ratio_aligned[i] > 1.8 and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla L3 level AND ATR ratio > 1.8 AND bearish 1d trend
            elif curr_close < l3_aligned[i] and atr_ratio_aligned[i] > 1.8 and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATR_VolumeSpike_1dEMA34_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
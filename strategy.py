#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and 1d ATR volume spike filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA34 trend filter, 1d for ATR volume confirmation.
- Entry: Long when price breaks above Camarilla H3 AND ATR ratio > 1.5 AND price > 4h EMA34.
         Short when price breaks below Camarilla L3 AND ATR ratio > 1.5 AND price < 4h EMA34.
- Exit: Opposite Camarilla breakout (L3 for longs, H3 for shorts) OR price crosses 4h EMA34 in opposite direction.
- Signal size: 0.20 discrete to minimize fee drag while maintaining profit potential.
- Uses Camarilla pivot points (H3/L3) as dynamic support/resistance levels proven effective in crypto.
- ATR ratio (current ATR/20-period ATR) > 1.5 confirms volatility expansion to avoid false breakouts.
- 4h EMA34 provides trend filter to avoid counter-trend trades in ranging markets.
- Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
- Session filter: 08-20 UTC to reduce noise trades during low-volume periods.
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

def camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot points (H3, L3 levels)."""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    h3 = pivot + (range_val * 1.1 / 4.0)
    l3 = pivot - (range_val * 1.1 / 4.0)
    return h3, l3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h trend filter: EMA34
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 40:
        return np.zeros(n)
    
    ema34_4h = ema(df_4h['close'].values, 34)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivots for current bar using available data
        if i < 2:  # Need at least 2 bars for meaningful pivot calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        
        # Calculate Camarilla H3/L3 for current bar
        h3, l3 = camarilla_pivots(curr_high, curr_low, curr_close)
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 4h EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below Camarilla L3 OR price falls below 4h EMA34
            if position == 1:
                if curr_close < l3 or curr_close < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Camarilla H3 OR price rises above 4h EMA34
            elif position == -1:
                if curr_close > h3 or curr_close > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above Camarilla H3 AND ATR ratio > 1.5 AND bullish 4h trend
            if curr_close > h3 and atr_ratio_aligned[i] > 1.5 and curr_close > ema34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla L3 AND ATR ratio > 1.5 AND bearish 4h trend
            elif curr_close < l3 and atr_ratio_aligned[i] > 1.5 and curr_close < ema34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_Trend_1dATR_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0
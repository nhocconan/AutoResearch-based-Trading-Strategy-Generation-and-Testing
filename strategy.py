#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and 1d volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter and ATR-based volume spike.
- Entry: Long when price breaks above Camarilla R1 AND ATR ratio > 1.8 AND price > 1d EMA34.
         Short when price breaks below Camarilla S1 AND ATR ratio > 1.8 AND price < 1d EMA34.
- Exit: Opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Uses proven Camarilla structure from top performers with strict filters to avoid overtrading.
- Works in bull markets (buy R1 breakouts in uptrend) and bear markets (sell S1 breakdowns in downtrend).
- Estimated trades: ~100 total over 4 years (~25/year) based on strict confluence requirements.
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
    """Calculate Camarilla pivot levels (R1, S1)."""
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d, additional_delay_bars=1)
    
    # Calculate 1d ATR for volume spike filter (current ATR / 20-period ATR)
    atr_20 = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    atr_current = atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 1)
    atr_ratio = atr_current / (atr_20 + 1e-10)  # Avoid division by zero
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Calculate Camarilla levels for previous day (using previous bar's data)
        if i >= 1:
            phigh = high[i-1]
            plow = low[i-1]
            pclose = close[i-1]
            r1, s1 = camarilla_pivots(phigh, plow, pclose)
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Exit conditions: opposite Camarilla breakout OR price crosses 1d EMA34 in opposite direction
        if position != 0:
            # Exit long: price breaks below S1 OR price falls below 1d EMA34
            if position == 1:
                if curr_low < s1 or curr_close < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above R1 OR price rises above 1d EMA34
            elif position == -1:
                if curr_high > r1 or curr_close > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volatility confirmation and trend filter
        if position == 0:
            # Long: price breaks above R1 AND ATR ratio > 1.8 AND bullish 1d trend
            if curr_high > r1 and atr_ratio_aligned[i] > 1.8 and curr_close > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND ATR ratio > 1.8 AND bearish 1d trend
            elif curr_low < s1 and atr_ratio_aligned[i] > 1.8 and curr_close < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dATR_VolumeSpike_1dEMA34_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0
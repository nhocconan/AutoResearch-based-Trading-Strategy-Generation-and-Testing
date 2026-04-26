#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_ATRStop
Hypothesis: Trade Camarilla R3/S3 breakouts with 1d EMA34 trend filter and ATR-based trailing stop.
- R3/S3 levels provide stronger breakout signals than R1/S1, reducing false signals
- 1d EMA34 ensures we trade with the higher timeframe trend to avoid counter-trend whipsaws
- ATR trailing stop locks in profits and limits drawdowns
- Target frequency: 20-40 trades/year on 4h to minimize fee drag
- Works in both bull and bear markets by following the 1d trend direction
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34)
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivot levels on 4h chart using previous bar's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels (R3) and Support levels (S3)
    R3 = pivot + (range_hl * 1.1 / 4.0)
    S3 = pivot - (range_hl * 1.1 / 4.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track highest high since entry for longs, lowest low since entry for shorts
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Start after warmup (need 34 for 1d EMA, 14 for ATR)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(trend_1d[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            # Update tracking variables
            if position == 1:
                highest_since_entry[i] = max(high[i], highest_since_entry[i-1]) if i > 0 else high[i]
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else low[i]
            elif position == -1:
                lowest_since_entry[i] = min(low[i], lowest_since_entry[i-1]) if i > 0 else low[i]
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else high[i]
            else:
                highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else high[i]
                lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else low[i]
            continue
        
        # Initialize tracking variables for current bar
        if i == start_idx:
            highest_since_entry[i] = high[i]
            lowest_since_entry[i] = low[i]
        else:
            highest_since_entry[i] = highest_since_entry[i-1]
            lowest_since_entry[i] = lowest_since_entry[i-1]
        
        # Update tracking variables based on position
        if position == 1:
            highest_since_entry[i] = max(high[i], highest_since_entry[i-1])
            lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == -1:
            lowest_since_entry[i] = min(low[i], lowest_since_entry[i-1])
            highest_since_entry[i] = highest_since_entry[i-1]
        
        # ATR-based stoploss conditions
        stop_long = False
        stop_short = False
        if position == 1 and highest_since_entry[i] > 0:
            stop_long = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
        elif position == -1 and lowest_since_entry[i] > 0:
            stop_short = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
        
        # Camarilla R3/S3 breakout conditions with trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1d uptrend
            if close[i] > R3[i] and trend_1d[i] == 1:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
            # Short: Price breaks below Camarilla S3 AND 1d downtrend
            elif close[i] < S3[i] and trend_1d[i] == -1:
                signals[i] = -0.25
                position = -1
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 1d trend turns down OR ATR stoploss hit
            if close[i] < S3[i] or trend_1d[i] == -1 or stop_long:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 1d trend turns up OR ATR stoploss hit
            if close[i] > R3[i] or trend_1d[i] == 1 or stop_short:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_ATRStop"
timeframe = "4h"
leverage = 1.0
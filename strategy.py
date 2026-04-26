#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: 12h Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla R1 level AND 1w EMA50 uptrend AND volume > 2.0 * volume_ma(20)
- Short when price breaks below Camarilla S1 level AND 1w EMA50 downtrend AND volume > 2.0 * volume_ma(20)
- Uses Camarilla pivot levels from 12h chart for structure-based breakouts
- 1w EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Designed for low frequency (target 12-37 trades/year on 12h) to minimize fee drag
- Works in both bull and bear markets by aligning with 1w trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter (needs completed 1w candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
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
    
    # Calculate Camarilla pivot levels on 12h chart (primary timeframe)
    # Using previous bar's OHLC for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels (R1, R3)
    R1 = pivot + (range_hl * 1.1 / 12.0)
    R3 = pivot + (range_hl * 1.1 / 4.0)
    # Support levels (S1, S3)
    S1 = pivot - (range_hl * 1.1 / 12.0)
    S3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track highest high since entry for longs, lowest low since entry for shorts
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Start after warmup (need 50 for 1w EMA, 20 for volume MA, 14 for ATR)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(trend_1w[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Camarilla R1/S1 breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 1w uptrend AND volume spike
            if close[i] > R1[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
            # Short: Price breaks below Camarilla S1 AND 1w downtrend AND volume spike
            elif close[i] < S1[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 1w trend turns down OR ATR stoploss hit
            if close[i] < S3[i] or trend_1w[i] == -1 or stop_long:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 1w trend turns up OR ATR stoploss hit
            if close[i] > R3[i] or trend_1w[i] == 1 or stop_short:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
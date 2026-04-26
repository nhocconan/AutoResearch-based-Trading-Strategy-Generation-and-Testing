#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2
Hypothesis: Tighter Camarilla R1/S1 breakout with stronger volume confirmation (2.0x) and stricter trend filter (EMA50) to reduce trade count and fee drag.
- Long when price breaks above Camarilla R1 AND 1d EMA50 uptrend AND volume > 2.0 * volume_ma(20)
- Short when price breaks below Camarilla S1 AND 1d EMA50 downtrend AND volume > 2.0 * volume_ma(20)
- ATR-based stoploss: exit long if price < highest_high_since_entry - 2.5 * ATR(14)
- ATR-based stoploss: exit short if price > lowest_low_since_entry + 2.5 * ATR(14)
- Exit also on opposite Camarilla level touch (S3 for longs, R3 for shorts) or trend reversal
- Designed for lower frequency (target 15-30 trades/year on 4h) to minimize fee drag and improve test generalization
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (more stable than EMA34)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
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
    
    # Calculate Camarilla pivot levels on 4h chart (primary timeframe)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels (R1, R3)
    R1 = pivot + (range_hl * 1.1 / 12.0)
    R3 = pivot + (range_hl * 1.1 / 4.0)
    # Support levels (S1, S3)
    S1 = pivot - (range_hl * 1.1 / 12.0)
    S3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for stronger confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    # Track highest high since entry for longs, lowest low since entry for shorts
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    # Start after warmup (need 50 for 1d EMA, 20 for volume MA, 14 for ATR)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
            # Long: Price breaks above Camarilla R1 AND 1d uptrend AND volume spike
            if close[i] > R1[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]
                lowest_since_entry[i] = low[i]
            # Short: Price breaks below Camarilla S1 AND 1d downtrend AND volume spike
            elif close[i] < S1[i] and trend_1d[i] == -1 and volume_spike[i]:
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0
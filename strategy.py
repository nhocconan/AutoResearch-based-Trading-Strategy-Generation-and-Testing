#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h volume spike and 1d ATR volatility filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for volume confirmation, 1d for ATR filter and Camarilla pivot levels.
- Camarilla Pivots: R1, S1 levels from prior 1d OHLC for breakout logic.
- Volume Filter: Current 4h volume > 1.5 * 20-period average 4h volume (avoid low-vol fakeouts).
- ATR Filter: Current ATR(14) < 1.8 * 20-period average ATR(14) on 1d to avoid extreme volatility.
- Entry: Long when close > R1 AND volume confirmation AND ATR filter.
         Short when close < S1 AND volume confirmation AND ATR filter.
- Exit: Opposite Camarilla break (long exits when close < S1, short exits when close > R1).
- Signal size: 0.20 discrete to minimize fee drag.
- Designed to capture momentum bursts in both bull and bear markets while filtering chop/whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivots (R1, S1) from prior day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Shifted to avoid look-ahead
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels (using standard formula)
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (waits for 1d bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h volume average for confirmation (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # Calculate 1d ATR(14) and its 20-period average for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0  # First bar has no previous close
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20_1d)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for volume/ATR MA, 14 for ATR
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_ma_20_4h_val = vol_ma_20_4h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_ma_20_1d_val = atr_ma_20_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average 4h volume
        # We approximate current 4h volume by using current 1h volume (conservative)
        volume_confirm = curr_volume > 1.5 * vol_ma_20_4h_val
        
        # ATR filter: current 1d ATR < 1.8 * 20-period average ATR (avoid extreme volatility)
        atr_filter = atr_1d_val < 1.8 * atr_ma_20_1d_val
        
        # Camarilla breakout conditions
        broke_above_r1 = curr_close > r1_level
        broke_below_s1 = curr_close < s1_level
        
        # Exit conditions: opposite Camarilla break
        if position != 0:
            # Exit long: close breaks below S1
            if position == 1:
                if curr_close < s1_level:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above R1
            elif position == -1:
                if curr_close > r1_level:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with volume and ATR filters
        if position == 0:
            # Long: break above R1 AND volume confirmation AND ATR filter
            long_condition = broke_above_r1 and volume_confirm and atr_filter
            
            # Short: break below S1 AND volume confirmation AND ATR filter
            short_condition = broke_below_s1 and volume_confirm and atr_filter
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hVolSpike_1dATRFilter_v1"
timeframe = "1h"
leverage = 1.0
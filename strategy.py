#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend filter and 1d for Camarilla pivot levels.
- Camarilla Pivots: R1, S1 levels from prior 1d OHLC for breakout logic.
- Trend Filter: Price > 12h EMA50 for long bias, Price < 12h EMA50 for short bias.
- Volume Filter: Current 4h volume > 2.0 * 20-period average 4h volume (avoid low-vol fakeouts).
- Entry: Long when close > R1 AND price > 12h EMA50 AND volume confirmation.
         Short when close < S1 AND price < 12h EMA50 AND volume confirmation.
- Exit: Opposite Camarilla break (long exits when close < S1, short exits when close > R1).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture momentum bursts in both bull and bear markets while filtering chop/whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
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
    
    # Camarilla R1 and S1 levels
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (waits for 1d bar close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for volume MA, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_50_level = ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20[i]
        
        # Trend filter: price above/below 12h EMA50
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
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
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: break above R1 AND above 12h EMA50 AND volume confirmation
            long_condition = broke_above_r1 and above_ema and volume_confirm
            
            # Short: break below S1 AND below 12h EMA50 AND volume confirmation
            short_condition = broke_below_s1 and below_ema and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0
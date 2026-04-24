#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h for EMA50 trend direction and volume spike filter.
- Camarilla pivot levels calculated from prior 1d OHLC: R1, S1 as entry levels.
- Trend Filter: Price > EMA50(12h) for long bias, Price < EMA50(12h) for short bias.
- Volume Confirmation: Current volume > 1.5 * 20-period average volume on 4h.
- Entry: Long when close crosses above R1 AND long bias AND volume confirmation.
         Short when close crosses below S1 AND short bias AND volume confirmation.
- Exit: Opposite Camarilla level (long exits at S1, short exits at R1) or reverse signal.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 12h trend and fading extremes only with volume confirmation.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h volume average for confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate prior 1d OHLC for Camarilla levels (need 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get prior 1day OHLC (shifted by 1 to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12
    
    # Align 1d Camarilla levels to 4h timeframe (completed 1d bar only)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        prev_close = close[i-1] if i > 0 else curr_close
        
        # Trend filter: price > EMA50 for long bias, price < EMA50 for short bias
        long_bias = curr_close > ema50_12h_aligned[i]
        short_bias = curr_close < ema50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
        
        # Camarilla breakouts
        crossed_above_r1 = (prev_close <= curr_r1) and (curr_close > curr_r1)
        crossed_below_s1 = (prev_close >= curr_s1) and (curr_close < curr_s1)
        
        # Exit conditions: opposite Camarilla level or reverse signal
        if position != 0:
            # Exit long: price crosses below S1
            if position == 1:
                if curr_close < curr_s1:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above R1
            elif position == -1:
                if curr_close > curr_r1:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: close crosses above R1 AND long bias AND volume confirmation
            long_condition = crossed_above_r1 and long_bias and volume_confirm
            
            # Short: close crosses below S1 AND short bias AND volume confirmation
            short_condition = crossed_below_s1 and short_bias and volume_confirm
            
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

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0
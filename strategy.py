#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend direction (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Camarilla pivot levels from 1d: R3/S3 for mean reversion fades, R4/S4 for breakout continuation.
- Entry: Long when price > R4 AND 1w uptrend AND volume > 1.5 * 20-period average volume.
         Short when price < S4 AND 1w downtrend AND volume > 1.5 * 20-period average volume.
         Mean reversion longs at S3 with 1w uptrend, shorts at R3 with 1w downtrend.
- Exit: Opposite Camarilla level (R3 for long exit from S3, S3 for short exit from R3;
        R4 for long exit from R4 breakout, S4 for short exit from S4 breakout).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via R4/S4 breakouts and in bear markets via R3/S3 fades.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 1 day for pivot calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculation (based on previous day)
    # We'll use rolling window to get previous day's OHLC
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Shift to get previous day's values
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + range_hl * 1.1 / 4.0
    r4 = pivot + range_hl * 1.1 / 2.0
    s3 = pivot - range_hl * 1.1 / 4.0
    s4 = pivot - range_hl * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema50_1w_aligned[i]
        downtrend = curr_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions
        if position != 0:
            exit_signal = False
            
            # Exit long positions
            if position == 1:
                # Exit long from S3 mean reversion: price > R3
                if curr_close > r3_aligned[i]:
                    exit_signal = True
                # Exit long from R4 breakout: price < R4
                elif curr_close < r4_aligned[i]:
                    exit_signal = True
                    
            # Exit short positions
            elif position == -1:
                # Exit short from R3 mean reversion: price < R3
                if curr_close < r3_aligned[i]:
                    exit_signal = True
                # Exit short from S4 breakout: price > S4
                elif curr_close > s4_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        if position == 0:
            # Mean reversion entries (fade at S3/R3)
            long_mr = (curr_close < s3_aligned[i] and 
                      uptrend and
                      volume_confirm)
            
            short_mr = (curr_close > r3_aligned[i] and 
                       downtrend and
                       volume_confirm)
            
            # Breakout entries (break at R4/S4)
            long_breakout = (curr_close > r4_aligned[i] and 
                            uptrend and
                            volume_confirm)
            
            short_breakout = (curr_close < s4_aligned[i] and 
                             downtrend and
                             volume_confirm)
            
            if long_mr or long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_mr or short_breakout:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3R4S4_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
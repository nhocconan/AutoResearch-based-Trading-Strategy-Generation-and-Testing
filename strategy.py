#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_WeeklyPivotDirection
Hypothesis: On 6h timeframe, price breaking Camarilla R3/S3 levels with weekly pivot direction alignment and volume confirmation provides robust breakout signals. Weekly pivot (calculated from prior week OHLC) offers structural bias that works in both bull and bear markets by filtering breakouts against the higher timeframe trend. Volume confirmation ensures momentum validity. Targets 12-37 trades/year (~50-150 over 4 years) to stay within optimal trade frequency for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week OHLC
    # Use prior week's high, low, close to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    # Weekly trend: price above/below weekly pivot
    weekly_trend_up = weekly_pivot > 0  # placeholder, will be replaced with aligned version
    weekly_trend_down = weekly_pivot > 0  # placeholder
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Camarilla levels from previous 6h bar
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of weekly pivot calculation (need 1 week), ATR(14), volume MA(20)
    start_idx = max(20, 14, 20) + 1  # weekly data aligned via mtf_data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 1.8  # volume at least 1.8x average
        
        # Weekly trend direction: price above/below weekly pivot
        weekly_trend_up = close_val > weekly_pivot_aligned[i]
        weekly_trend_down = close_val < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND weekly trend up AND volume confirmation
            long_signal = (close_val > camarilla_r3[i]) and weekly_trend_up and vol_confirmed
            
            # Short: price breaks below Camarilla S3 AND weekly trend down AND volume confirmation
            short_signal = (close_val < camarilla_s3[i]) and weekly_trend_down and vol_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: weekly trend flips down OR price hits ATR stoploss
            if (not weekly_trend_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: weekly trend flips up OR price hits ATR stoploss
            if (not weekly_trend_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_WeeklyPivotDirection"
timeframe = "6h"
leverage = 1.0
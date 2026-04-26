#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyPivotFilter_v1
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakout with 1d EMA50 trend filter and weekly pivot direction filter. Weekly pivot adds structural bias from higher timeframe (1w) to avoid counter-trend trades in strong weekly trends. Designed for low trade frequency (target 12-35/year) to minimize fee drag. Works in bull markets (breakouts with 1d/1w trend alignment) and bear markets (fade at extremes when weekly pivot opposes breakout). Uses discrete position sizing (0.25) to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot from previous 1w bar
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high_1w = np.where(np.isnan(prev_high_1w), df_1w['high'].values, prev_high_1w)
    prev_low_1w = np.where(np.isnan(prev_low_1w), df_1w['low'].values, prev_low_1w)
    prev_close_1w = np.where(np.isnan(prev_close_1w), df_1w['close'].values, prev_close_1w)
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate ATR(14) for stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 4.0)   # R3 level
    s3 = pivot - (range_hl * 1.1 / 4.0)   # S3 level
    
    # Align Camarilla levels to 6h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(50), volume MA, ATR
    start_idx = max(50, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        weekly_pivot_bias_up = close_val > pivot_1w_aligned[i]  # Above weekly pivot = bullish bias
        weekly_pivot_bias_down = close_val < pivot_1w_aligned[i]  # Below weekly pivot = bearish bias
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R3 AND 1d trend up AND above weekly pivot AND volume confirmation
            long_signal = (close_val > r3_aligned[i]) and trend_1d_up and weekly_pivot_bias_up and vol_confirm
            
            # Short: price breaks below S3 AND 1d trend down AND below weekly pivot AND volume confirmation
            short_signal = (close_val < s3_aligned[i]) and trend_1d_down and weekly_pivot_bias_down and vol_confirm
            
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
            # Exit: trend flips down OR price hits ATR stoploss
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend flips up OR price hits ATR stoploss
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_WeeklyPivotFilter_v1"
timeframe = "6h"
leverage = 1.0
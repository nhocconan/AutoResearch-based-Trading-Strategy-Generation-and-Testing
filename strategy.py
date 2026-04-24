#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter (bullish if price > EMA50, bearish if price < EMA50).
- Camarilla levels from 1d: R3, S3, R4, S4 calculated from prior 1d OHLC.
- Entry: Long when price > R3 AND bullish 1d trend AND volume > 1.5 * 20-period average volume.
         Short when price < S3 AND bearish 1d trend AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla breakout (price < S3 for long exit, price > R3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 1d trend, avoiding counter-trend whipsaws.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate prior 1d OHLC for Camarilla levels (shifted by 1 to avoid look-ahead)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3, S3, R4, S4
    # R3 = close_prev + 1.1 * (high_prev - low_prev) / 4
    # S3 = close_prev - 1.1 * (high_prev - low_prev) / 4
    # R4 = close_prev + 1.1 * (high_prev - low_prev) / 2
    # S4 = close_prev - 1.1 * (high_prev - low_prev) / 2
    camarilla_calc = (high_prev - low_prev)
    r3 = close_prev + 1.1 * camarilla_calc / 4
    s3 = close_prev - 1.1 * camarilla_calc / 4
    r4 = close_prev + 1.1 * camarilla_calc / 2
    s4 = close_prev - 1.1 * camarilla_calc / 2
    
    # Align HTF indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: bullish if price > EMA50, bearish if price < EMA50
        bullish_trend = curr_close > ema50_aligned[i]
        bearish_trend = curr_close < ema50_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla breakout (exit at S3/R3)
        if position != 0:
            # Exit long: price < S3
            if position == 1:
                if curr_close < s3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > R3
            elif position == -1:
                if curr_close > r3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with trend and volume filters
        if position == 0:
            # Long: price > R3 AND bullish trend AND volume confirmation
            long_condition = (curr_close > r3_aligned[i] and 
                            bullish_trend and
                            volume_confirm)
            
            # Short: price < S3 AND bearish trend AND volume confirmation
            short_condition = (curr_close < s3_aligned[i] and 
                             bearish_trend and
                             volume_confirm)
            
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

name = "6h_Camarilla_R3S3_Breakout_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0
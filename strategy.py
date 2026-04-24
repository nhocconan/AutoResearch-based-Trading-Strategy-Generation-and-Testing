#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla levels (R3, S3, R4, S4) calculated from prior 1d OHLC.
- Entry: Long when price > R3 AND bullish 1d trend AND volume > 1.5 * 20-period average volume.
         Short when price < S3 AND bearish 1d trend AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Camarilla level (price < S3 for long exit, price > R3 for short exit) OR
        trend reversal (price crosses 1d EMA50 in opposite direction).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong intraday moves aligned with daily trend, avoiding counter-trend whipsaws.
- Works in bull markets via longs, bear markets via shorts, and ranges via reduced frequency.
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
    
    # Align EMA50 to 6h timeframe (trend from prior completed 1d candle)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate prior 1d OHLC for Camarilla levels (using completed 1d candles)
    # We need to shift by 1 to avoid look-ahead: use previous day's OHLC for today's levels
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior completed 1d candle OHLC (shifted by 1)
    high_1d_prev = df_1d['high'].shift(1).values
    low_1d_prev = df_1d['low'].shift(1).values
    close_1d_prev = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels from prior 1d candle
    range_1d = high_1d_prev - low_1d_prev
    camarilla_multiplier = range_1d * 1.1 / 12  # Standard Camarilla multiplier
    
    # R3, S3, R4, S4 levels
    r3 = close_1d_prev + camarilla_multiplier * 3
    s3 = close_1d_prev - camarilla_multiplier * 3
    r4 = close_1d_prev + camarilla_multiplier * 4
    s4 = close_1d_prev - camarilla_multiplier * 4
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need 50 for EMA50, plus 1 for shift = 51, but we'll check NaN inside
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema50_1d_aligned[i]
        bearish_trend = curr_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions
        if position != 0:
            # Exit long: price < S3 OR trend turns bearish (price < EMA50)
            if position == 1:
                if curr_close < s3_aligned[i] or curr_close < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > R3 OR trend turns bullish (price > EMA50)
            elif position == -1:
                if curr_close > r3_aligned[i] or curr_close > ema50_1d_aligned[i]:
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
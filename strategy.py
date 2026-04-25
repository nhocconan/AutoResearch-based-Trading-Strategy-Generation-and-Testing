#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_RegimeFilter
Hypothesis: Camarilla R3/S3 breakouts on 6h timeframe with 1d EMA50 trend filter and 
6h ADX regime filter (ADX>25 for trending markets). Only trade breakouts in direction 
of daily trend when market is trending. Uses discrete position sizing (0.25) to minimize 
fee churn. Designed for low trade frequency (~15-25/year) to work in both bull and bear 
markets via trend alignment and regime filtering. Camarilla R3/S3 levels represent 
strong support/resistance where breakouts have higher follow-through than R1/S1 levels.
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
    
    # Get 1d data for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate previous day's Camarilla levels (use prior day's OHLC)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We need to shift by 1 to use previous day's levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First value will be invalid (rolled from last), but min_periods will handle it
    
    # Calculate Camarilla levels for previous day
    prev_range_1d = prev_high_1d - prev_low_1d
    camarilla_r3_1d = prev_close_1d + (prev_range_1d * 1.1 / 4)
    camarilla_s3_1d = prev_close_1d - (prev_range_1d * 1.1 / 4)
    
    # Align HTF indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d, additional_delay_bars=1)
    
    # Calculate 6h ADX for regime filter (trending when ADX > 25)
    # ADX calculation: +DI, -DI, DX, then ADX = smoothed DX
    period = 14
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original indices
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    def smooth_wilder(arr, period):
        """Wilder's smoothing (EMA with alpha=1/period)"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smooth_wilder(tr, period)
    plus_di = 100 * smooth_wilder(plus_dm, period) / atr
    minus_di = 100 * smooth_wilder(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_wilder(dx, period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and ADX (2*period)
    start_idx = max(50, 2*period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx[i] > 25
        
        if position == 0:
            # Look for Camarilla R3/S3 breakout signals with trend filter
            # Long: price breaks above camarilla R3 in uptrend (close > EMA50)
            # Short: price breaks below camarilla S3 in downtrend (close < EMA50)
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema50_aligned[i]) and is_trending
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema50_aligned[i]) and is_trending
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below EMA50 (trend reversal) or breaks S3 (mean reversion)
            exit_signal = (close[i] < ema50_aligned[i]) or (close[i] < camarilla_s3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal) or breaks R3 (mean reversion)
            exit_signal = (close[i] > ema50_aligned[i]) or (close[i] > camarilla_r3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_RegimeFilter"
timeframe = "6h"
leverage = 1.0
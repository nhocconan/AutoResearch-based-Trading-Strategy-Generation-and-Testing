#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_ADXFilter_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 6h with 1d trend (EMA50) and ADX filter.
Only take breakouts when 1d ADX > 25 (trending market) to avoid false breakouts in ranging markets.
In bull markets (price > 1d EMA50): long on break above R3, short on break below S3.
In bear markets (price < 1d EMA50): short on break below S3, long on break above R3.
Exit on opposite Camarilla level (R3/S3) or trend reversal.
Position size: 0.25 to limit drawdown.
Target: 15-30 trades/year to stay well under 300-trade 6h hard max.
Works in both bull and bear markets by using 1d trend filter and ADX to confirm regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 bars for ADX
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed arrays
    tr_smooth = np.full_like(tr, np.nan)
    plus_dm_smooth = np.full_like(plus_dm, np.nan)
    minus_dm_smooth = np.full_like(minus_dm, np.nan)
    
    # First smoothed value is simple average
    if len(tr) >= period:
        tr_smooth[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        # Subsequent values using Wilder's smoothing
        for i in range(period + 1, len(tr)):
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # Directional Indicators
    plus_di = np.full_like(tr, np.nan)
    minus_di = np.full_like(tr, np.nan)
    dx = np.full_like(tr, np.nan)
    
    valid_mask = ~np.isnan(tr_smooth) & (tr_smooth != 0)
    plus_di[valid_mask] = (plus_dm_smooth[valid_mask] / tr_smooth[valid_mask]) * 100
    minus_di[valid_mask] = (minus_dm_smooth[valid_mask] / tr_smooth[valid_mask]) * 100
    
    dx_mask = ~np.isnan(plus_di) & ~np.isnan(minus_di) & ((plus_di + minus_di) != 0)
    dx[dx_mask] = (np.abs(plus_di[dx_mask] - minus_di[dx_mask]) / (plus_di[dx_mask] + minus_di[dx_mask])) * 100
    
    # ADX is smoothed DX
    adx = np.full_like(dx, np.nan)
    if len(dx) >= period:
        # First ADX value is simple average of DX
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        # Subsequent values using Wilder's smoothing
        for i in range(2*period, len(dx)):
            if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            else:
                adx[i] = np.nan
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    # Camarilla levels are based on previous day's range
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_r4 = np.full_like(close_1d, np.nan)
    camarilla_s4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        if not np.isnan(prev_high[i]) and not np.isnan(prev_low[i]) and not np.isnan(prev_close[i]):
            range_val = prev_high[i] - prev_low[i]
            camarilla_r3[i] = prev_close[i] + range_val * 1.1 / 4
            camarilla_s3[i] = prev_close[i] - range_val * 1.1 / 4
            camarilla_r4[i] = prev_close[i] + range_val * 1.1 / 2
            camarilla_s4[i] = prev_close[i] - range_val * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and ADX (2*14=28)
    start_idx = max(50, 28)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # ADX filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long setup: price breaks above R3 + 1d uptrend + strong trend
            long_setup = (close[i] > camarilla_r3_aligned[i]) and htf_1d_bullish and strong_trend
            
            # Short setup: price breaks below S3 + 1d downtrend + strong trend
            short_setup = (close[i] < camarilla_s3_aligned[i]) and htf_1d_bearish and strong_trend
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches S3 (stop) OR 1d trend turns bearish OR trend weakens
            if (close[i] <= camarilla_s3_aligned[i]) or (not htf_1d_bullish) or (not strong_trend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R3 (stop) OR 1d trend turns bullish OR trend weakens
            if (close[i] >= camarilla_r3_aligned[i]) or (htf_1d_bullish) or (not strong_trend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_ADXFilter_v1"
timeframe = "6h"
leverage = 1.0
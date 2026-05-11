#!/usr/bin/env python3
"""
1d_1w_RangeBreakout_Volume_Confirmation
Hypothesis: Breakout of 1-week high/low on 1d timeframe with volume confirmation and ADX trend filter.
- Long when: price breaks above 1w high + volume > 20-period average + ADX > 25 (trending)
- Short when: price breaks below 1w low + volume > 20-period average + ADX > 25 (trending)
- Exit when price returns to 1w midpoint (mean reversion within the weekly range)
Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Uses ADX filter to avoid false breakouts in ranging markets and focus on true momentum.
Works in bull (breakouts up) and bear (breakouts down) markets.
"""

name = "1d_1w_RangeBreakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for range calculation and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:  # Need at least 14 periods for ADX
        return np.zeros(n)
    
    # 1d OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- 1w High/Low for breakout ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Use previous week's high/low to avoid look-ahead
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_high_1w[0] = np.nan  # First value has no previous week
    prev_low_1w[0] = np.nan
    
    # Align to 1d timeframe (previous week's high/low)
    high_1w_align = align_htf_to_ltf(prices, df_1w, prev_high_1w)
    low_1w_align = align_htf_to_ltf(prices, df_1w, prev_low_1w)
    
    # --- Volume Confirmation: 1d volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # --- ADX Calculation on 1w ---
    high_1w_arr = df_1w['high'].values
    low_1w_arr = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # True Range
    tr1 = high_1w_arr[1:] - low_1w_arr[1:]
    tr2 = np.abs(high_1w_arr[1:] - close_1w_arr[:-1])
    tr3 = np.abs(low_1w_arr[1:] - close_1w_arr[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    up_move = high_1w_arr[1:] - high_1w_arr[:-1]
    down_move = low_1w_arr[:-1] - low_1w_arr[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])  # Skip first NaN
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
            else:
                result[i] = np.nan
        return result
    
    atr_1w = wilders_smoothing(tr, 14)
    plus_di_1w = wilders_smoothing(plus_dm, 14)
    minus_di_1w = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di = np.where(atr_1w != 0, (plus_di_1w / atr_1w) * 100, 0)
    minus_di = np.where(atr_1w != 0, (minus_di_1w / atr_1w) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx_1w = wilders_smoothing(dx, 14)
    
    # Align ADX to 1d timeframe
    adx_1w_align = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # --- 1w Midpoint for exit ---
    midpoint_1w = (prev_high_1w + prev_low_1w) / 2.0
    midpoint_1w_align = align_htf_to_ltf(prices, df_1w, midpoint_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_1w_align[i]) or np.isnan(low_1w_align[i]) or 
            np.isnan(midpoint_1w_align[i]) or np.isnan(adx_1w_align[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume_1d[i] > vol_ma_20[i]
        
        # ADX trend filter (>25 indicates trending market)
        adx_ok = adx_1w_align[i] > 25
        
        if position == 0:
            # Look for breakouts in trending markets with volume
            if close_1d[i] > high_1w_align[i] and vol_ok and adx_ok:
                # Long breakout above weekly high
                signals[i] = 0.25
                position = 1
            elif close_1d[i] < low_1w_align[i] and vol_ok and adx_ok:
                # Short breakout below weekly low
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to weekly midpoint (mean reversion within range)
            if position == 1:
                # Exit long: price returns to or below midpoint
                if close_1d[i] <= midpoint_1w_align[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to or above midpoint
                if close_1d[i] >= midpoint_1w_align[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
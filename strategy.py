#!/usr/bin/env python3
"""
6h Weekly Donchian Breakout + Volume Spike + 1d ADX Trend Filter
Hypothesis: Weekly Donchian channels (20-period) capture major support/resistance.
Breakouts with volume confirmation and 1d ADX > 25 filter trade only in strong trends,
avoiding whipsaws in ranging markets. Works in bull/bear via ADX trend strength filter.
Designed for 50-150 trades over 4 years with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr

def calculate_adx(high, low, close, period):
    """Calculate Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+, DM-
    tr_period = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    dm_plus_period = pd.Series(dm_plus).rolling(window=period, min_periods=period).mean().values
    dm_minus_period = pd.Series(dm_minus).rolling(window=period, min_periods=period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_period / tr_period
    di_minus = 100 * dm_minus_period / tr_period
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX trend filter
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1w data for weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Weekly Donchian channels (20-period)
    weekly_dc_upper = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_dc_lower = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    weekly_dc_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_dc_upper)
    weekly_dc_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_dc_lower)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for trailing stop (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for ADX, Donchian, volume MA, ATR
    start_idx = max(34, 20, 20, 14) + 5  # ADX(14) needs ~28, Donchian(20) needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(weekly_dc_upper_aligned[i]) or np.isnan(weekly_dc_lower_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        strong_trend = adx_1d_aligned[i] > 25  # ADX > 25 indicates strong trend
        
        # Breakout conditions using weekly Donchian channels
        breakout_long = curr_close > weekly_dc_upper_aligned[i]
        breakout_short = curr_close < weekly_dc_lower_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Weekly Donchian breakout + volume spike + strong trend (ADX > 25)
            long_entry = breakout_long and vol_spike and strong_trend
            short_entry = breakout_short and vol_spike and strong_trend
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_high_since_entry = curr_high
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: update highest high and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to weekly Donchian lower, trend weakening, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if curr_close < weekly_dc_lower_aligned[i] or adx_1d_aligned[i] < 20 or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: retrace to weekly Donchian upper, trend weakening, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if curr_close > weekly_dc_upper_aligned[i] or adx_1d_aligned[i] < 20 or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyDonchian20_Breakout_VolumeSpike_1dADX25_TrendFilter_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h Weekly Volume Profile POC + 1d EMA50 Trend + ATR Stop
Hypothesis: Weekly Point of Control (POC) from volume profile acts as strong support/resistance.
Price retesting weekly POC with volume confirmation and 1d EMA50 trend alignment offers
high-probability entries in both bull and bear markets. Weekly POC calculated from
prior week's TPO (30-min bins) as proxy for volume profile. Designed for 50-150 trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = calculate_ema(df_1d['close'].values, 50)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w data for weekly POC (Point of Control) - proxy using VWAP of weekly OHLC
    df_1w = get_htf_data(prices, '1w')
    # Weekly VWAP as POC proxy: typical price * volume summed / volume summed
    weekly_typical = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    weekly_volume = df_1w['volume'].values
    # Calculate cumulative VWAP-like POC for each week
    weekly_poc = np.full_like(df_1w['close'].values, np.nan)
    for i in range(len(df_1w)):
        if i == 0 or pd.isna(weekly_volume[i]) or weekly_volume[i] == 0:
            weekly_poc[i] = weekly_typical[i]
        else:
            # Simplified: use weekly typical price as POC (reasonable proxy)
            weekly_poc[i] = weekly_typical[i]
    # Shift to use prior week's POC
    weekly_poc_shifted = np.roll(weekly_poc, 1)
    weekly_poc_shifted[0] = np.nan
    
    # Weekly high/low for context
    weekly_high = np.roll(df_1w['high'].values, 1)
    weekly_low = np.roll(df_1w['low'].values, 1)
    weekly_high[0] = np.nan
    weekly_low[0] = np.nan
    
    # Align weekly POC and levels
    weekly_poc_aligned = align_htf_to_ltf(prices, df_1w, weekly_poc_shifted, additional_delay_bars=0)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high, additional_delay_bars=0)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low, additional_delay_bars=0)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # ATR for trailing stop (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Start index: need enough for EMA, volume MA, ATR
    start_idx = max(50, 20, 14) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(weekly_poc_aligned[i]) or np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Distance to weekly POC (normalized by ATR)
        poc_distance = abs(curr_close - weekly_poc_aligned[i]) / (atr[i] + 1e-10)
        
        # Rejection signals: price near weekly POC with volume spike
        near_poc = poc_distance < 0.5  # Within 0.5 ATR of weekly POC
        long_rejection = near_poc and curr_close > weekly_poc_aligned[i] and curr_low <= weekly_poc_aligned[i]
        short_rejection = near_poc and curr_close < weekly_poc_aligned[i] and curr_high >= weekly_poc_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Weekly POC rejection + volume spike + 1d EMA50 trend alignment
            long_entry = long_rejection and vol_spike and (curr_close > ema_50_1d_aligned[i])
            short_entry = short_rejection and vol_spike and (curr_close < ema_50_1d_aligned[i])
            
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
            
            # Exit conditions: price moves away from POC, trend change, or ATR trailing stop
            trailing_stop = highest_high_since_entry - 2.5 * atr[i]
            if curr_close < weekly_poc_aligned[i] or curr_close < ema_50_1d_aligned[i] or curr_close < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest low and check exit conditions
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            
            # Exit conditions: price moves away from POC, trend change, or ATR trailing stop
            trailing_stop = lowest_low_since_entry + 2.5 * atr[i]
            if curr_close > weekly_poc_aligned[i] or curr_close > ema_50_1d_aligned[i] or curr_close > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyVolumeProfile_POC_1dEMA50_Trend_VolumeSpike_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0
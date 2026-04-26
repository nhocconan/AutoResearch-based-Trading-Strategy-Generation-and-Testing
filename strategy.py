#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts only when aligned with weekly Camarilla pivot S3/R3 direction and confirmed by volume spike (>2.0x 20-period average). Use 1d EMA50 as trend filter. Discrete sizing: 0.25. Target: 15-30 trades/year to minimize fee drag while capturing strong momentum moves in both bull and bear markets via weekly structure and volume confirmation.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (UTC 8-20) for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for HTF trend and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for weekly Camarilla pivot (S3/R3) - HTF = 1w
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) on 6h for volume spike and breakout confirmation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly Camarilla levels from prior 1w bar (H1, L1, C1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    if len(high_1w) < 2:
        camarilla_r3 = np.full_like(close_1w, np.nan)
        camarilla_s3 = np.full_like(close_1w, np.nan)
    else:
        # Camarilla R3 = C + 1.1*(H-L)*1.1/4 = C + 1.1*(H-L)*0.275
        # Camarilla S3 = C - 1.1*(H-L)*1.1/4 = C - 1.1*(H-L)*0.275
        camarilla_r3 = close_1w[:-1] + 1.1 * (high_1w[:-1] - low_1w[:-1]) * 0.275
        camarilla_s3 = close_1w[:-1] - 1.1 * (high_1w[:-1] - low_1w[:-1]) * 0.275
        camarilla_r3 = np.concatenate([[np.nan], camarilla_r3])
        camarilla_s3 = np.concatenate([[np.nan], camarilla_s3])
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Volume average (20-period = ~5 days on 6h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start index: need warmup for calculations
    start_idx = max(20, 50, 14)  # Donchian, 1d EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            not in_session[i]):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_1d_val = ema_50_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above upper Donchian with weekly pivot R3 support (price > R3) and uptrend (close > EMA50) and volume confirmation
            long_signal = (close_val > upper_donchian) and (close_val > r3_val) and (close_val > ema_50_1d_val) and volume_confirmed
            # Short: price breaks below lower Donchian with weekly pivot S3 resistance (price < S3) and downtrend (close < EMA50) and volume confirmation
            short_signal = (close_val < lower_donchian) and (close_val < s3_val) and (close_val < ema_50_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR-based trailing stop: exit if price drops 2.5*ATR from high
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit conditions:
            # 1. Price breaks below weekly S3 (pivot support broken)
            elif close_val < s3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # 2. Trend reversal: close crosses below EMA50
            elif close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR-based trailing stop: exit if price rises 2.5*ATR from low
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit conditions:
            # 1. Price breaks above weekly R3 (pivot resistance broken)
            elif close_val > r3_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # 2. Trend reversal: close crosses above EMA50
            elif close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0
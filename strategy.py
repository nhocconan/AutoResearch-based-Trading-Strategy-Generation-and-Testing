#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla R4/S4 breakouts on 6h with 1-week EMA50 trend filter and volume confirmation (>2.0x 20-period average). R4/S4 represent strong breakout levels (vs R1/S1 which are mean-reversion). Weekly trend filter ensures we trade with the dominant higher-timeframe momentum, reducing whipsaws in both bull and bear markets. Volume confirmation ensures conviction. Discrete sizing 0.25 targets ~20-30 trades/year (80-120 total over 4 years) to minimize fee drag. Uses ATR(14) trailing stop (2.5x) for risk management.
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
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla calculation (R4/S4 from prior 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # ATR(14) on 6h for breakout confirmation and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Camarilla R4 and S4 from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    if len(high_1d) < 2:
        camarilla_r4 = np.full_like(close_1d_arr, np.nan)
        camarilla_s4 = np.full_like(close_1d_arr, np.nan)
    else:
        # Camarilla R4 = Close + 1.1*(High-Low)*1.1/2 = Close + 1.1*(High-Low)*0.55
        # Camarilla S4 = Close - 1.1*(High-Low)*1.1/2 = Close - 1.1*(High-Low)*0.55
        camarilla_r4 = close_1d_arr[:-1] + 1.1 * (high_1d[:-1] - low_1d[:-1]) * 0.55
        camarilla_s4 = close_1d_arr[:-1] - 1.1 * (high_1d[:-1] - low_1d[:-1]) * 0.55
        camarilla_r4 = np.concatenate([[np.nan], camarilla_r4])
        camarilla_s4 = np.concatenate([[np.nan], camarilla_s4])
    
    # Align Camarilla levels to 6h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of volume MA (20), 1w EMA (50), ATR (14)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict for quality)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        # Breakout threshold: price must close beyond Camarilla level by 2.0*ATR (strict)
        breakout_threshold = 2.0 * atr_val
        
        if position == 0:
            # Long: close above R4 + threshold, uptrend (close > EMA50_1w), volume confirmation
            long_signal = (close_val > r4_val + breakout_threshold) and (close_val > ema_50_1w_val) and volume_confirmed
            # Short: close below S4 - threshold, downtrend (close < EMA50_1w), volume confirmation
            short_signal = (close_val < s4_val - breakout_threshold) and (close_val < ema_50_1w_val) and volume_confirmed
            
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
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.5*ATR from high (wider stop for trend)
            if close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: price closes below S4 (mean reversion level)
            elif close_val < s4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: trend reversal (close below EMA50_1w)
            elif close_val < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.5*ATR from low
            if close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: price closes above R4 (mean reversion level)
            elif close_val > r4_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: trend reversal (close above EMA50_1w)
            elif close_val > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirmation
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume spike confirmation (>2.0x 20-day average). Uses discrete sizing (0.25) and ATR-based trailing stop (2.0x ATR). Target: 15-25 trades/year to avoid fee drag while capturing multi-day trends in BTC/ETH. Weekly trend ensures alignment with higher timeframe momentum, reducing whipsaw in bear markets.
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
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d for breakout confirmation and trailing stop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    high_1d_arr = df_1d['high'].values
    low_1d_arr = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    if len(high_1d_arr) < 2:
        camarilla_r1 = np.full_like(close_1d_arr, np.nan)
        camarilla_s1 = np.full_like(close_1d_arr, np.nan)
    else:
        camarilla_r1 = close_1d_arr[:-1] + 1.1 * (high_1d_arr[:-1] - low_1d_arr[:-1]) / 12
        camarilla_s1 = close_1d_arr[:-1] - 1.1 * (high_1d_arr[:-1] - low_1d_arr[:-1]) / 12
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels to 1d timeframe (no shift needed as already 1d)
    camarilla_r1_aligned = camarilla_r1  # Already aligned to 1d
    camarilla_s1_aligned = camarilla_s1  # Already aligned to 1d
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start index: need warmup for calculations
    start_idx = max(20, 50, 14)  # volume MA, 1w EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr_1d_aligned[i]) or
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
        ema_50_1w_val = ema_50_1w_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        # Breakout confirmation: price must close beyond Camarilla level by at least 1.5*ATR
        breakout_threshold = 1.5 * atr_val
        
        if position == 0:
            # Long: price closes above R1 + 1.5*ATR with uptrend (close > EMA50_1w) and volume confirmation
            long_signal = (close_val > r1_val + breakout_threshold) and (close_val > ema_50_1w_val) and volume_confirmed
            # Short: price closes below S1 - 1.5*ATR with downtrend (close < EMA50_1w) and volume confirmation
            short_signal = (close_val < s1_val - breakout_threshold) and (close_val < ema_50_1w_val) and volume_confirmed
            
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
            # ATR-based trailing stop: exit if price drops 2.0*ATR from high
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit conditions:
            # 1. Opposite breakout: price closes below S1 (exit long)
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # 2. Trend reversal: close crosses below EMA50_1w
            elif close_val < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR-based trailing stop: exit if price rises 2.0*ATR from low
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit conditions:
            # 1. Opposite breakout: price closes above R1 (exit short)
            elif close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # 2. Trend reversal: close crosses above EMA50_1w
            elif close_val > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0
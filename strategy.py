#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: Use 12h timeframe to reduce trade frequency and fee drag. Combine Camarilla R1/S1 breakouts from prior 1d with 1d EMA34 trend filter and volume spike (>2.0x 20-period average) for high-conviction entries. Discrete sizing 0.25. Target 12-30 trades/year to minimize fee drag while capturing strong trending moves in both bull and bear markets via 1d trend alignment.
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla R1 and S1 from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    if len(high_1d) < 2:
        camarilla_r1 = np.full_like(close_1d_arr, np.nan)
        camarilla_s1 = np.full_like(close_1d_arr, np.nan)
    else:
        camarilla_r1 = close_1d_arr[:-1] + 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_s1 = close_1d_arr[:-1] - 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels to 12h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of volume MA (20), 1d EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (stricter than before)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        # Breakout threshold: price must close beyond Camarilla level by 1.0*ATR (balanced for 12h)
        # Calculate ATR(14) on 12h for breakout confirmation
        if i >= 14:
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
            # Simplified ATR calculation for current bar (using rolling would need array)
            # We'll use a simplified approach: if we don't have ATR array, use price range
            atr_estimate = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        else:
            atr_estimate = 0.0
        breakout_threshold = 1.0 * atr_estimate
        
        if position == 0:
            # Long: close above R1 + threshold, uptrend (close > EMA34_1d), volume confirmation
            long_signal = (close_val > r1_val + breakout_threshold) and (close_val > ema_34_1d_val) and volume_confirmed
            # Short: close below S1 - threshold, downtrend (close < EMA34_1d), volume confirmation
            short_signal = (close_val < s1_val - breakout_threshold) and (close_val < ema_34_1d_val) and volume_confirmed
            
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
            # Exit: price closes below S1
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: trend reversal (close below EMA34_1d)
            elif close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Exit: price closes above R1
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: trend reversal (close above EMA34_1d)
            elif close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
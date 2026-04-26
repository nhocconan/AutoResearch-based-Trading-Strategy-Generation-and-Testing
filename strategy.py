#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirmation_v5
Hypothesis: Further tighten entry by requiring volume > 2.0x average AND price close beyond Camarilla level by 1.5*ATR(14). Use 1d EMA34 for trend filter and ATR-based trailing stop (2.5x ATR). Discrete position sizing: 0.25. Target: reduce trades to 15-25/year while maintaining edge in bull/bear via trend filter and volatility-adjusted breakout.
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
    
    # Get 1d data for HTF trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) on 4h for breakout confirmation and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
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
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume average (20-period = ~3.3 days on 4h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start index: need warmup for calculations
    start_idx = max(20, 34, 14)  # volume MA, 1d EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
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
        ema_34_1d_val = ema_34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        # Breakout confirmation: price must close beyond Camarilla level by at least 1.5*ATR
        breakout_threshold = 1.5 * atr_val
        
        if position == 0:
            # Long: price closes above R1 + 1.5*ATR with uptrend (close > EMA34) and volume confirmation
            long_signal = (close_val > r1_val + breakout_threshold) and (close_val > ema_34_1d_val) and volume_confirmed
            # Short: price closes below S1 - 1.5*ATR with downtrend (close < EMA34) and volume confirmation
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
            # 1. Opposite breakout: price closes below S1 (exit long)
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # 2. Trend reversal: close crosses below EMA34
            elif close_val < ema_34_1d_val:
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
            # 1. Opposite breakout: price closes above R1 (exit short)
            elif close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # 2. Trend reversal: close crosses above EMA34
            elif close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirmation_v5"
timeframe = "4h"
leverage = 1.0
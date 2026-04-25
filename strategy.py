#!/usr/bin/env python3
"""
6h Elder Ray Index with 12h Supertrend Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull/Bear Power) measures bull/bear strength relative to EMA13.
Combined with 12h Supertrend for regime filter and volume confirmation to avoid false signals.
Works in bull/bear via Supertrend regime filter. Discrete sizing (0.25) limits fee drag (~60-100 trades over 4 years).
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
    
    # Get 12h data for Supertrend regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0) for regime filter
    hl2_12h = (df_12h['high'] + df_12h['low']) / 2
    tr1_12h = df_12h['high'][1:] - df_12h['low'][1:]
    tr2_12h = np.abs(df_12h['high'][1:] - df_12h['close'][:-1])
    tr3_12h = np.abs(df_12h['low'][1:] - df_12h['close'][:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))])
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper_12h = hl2_12h + 3.0 * atr_12h
    lower_12h = hl2_12h - 3.0 * atr_12h
    supertrend_12h = np.full_like(hl2_12h, np.nan, dtype=float)
    direction_12h = np.ones_like(hl2_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2_12h)):
        if np.isnan(supertrend_12h[i-1]):
            supertrend_12h[i] = lower_12h[i]
            direction_12h[i] = 1
        else:
            if close_12h := df_12h['close'].iloc[i]:
                close_12h_val = df_12h['close'].iloc[i]
            else:
                close_12h_val = df_12h['close'].values[i]
            if direction_12h[i-1] == 1:
                supertrend_12h[i] = max(lower_12h[i], supertrend_12h[i-1])
                if close_12h_val < supertrend_12h[i]:
                    direction_12h[i] = -1
                    supertrend_12h[i] = upper_12h[i]
                else:
                    direction_12h[i] = 1
            else:
                supertrend_12h[i] = min(upper_12h[i], supertrend_12h[i-1])
                if close_12h_val > supertrend_12h[i]:
                    direction_12h[i] = 1
                    supertrend_12h[i] = lower_12h[i]
                else:
                    direction_12h[i] = -1
    
    supertrend_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h.astype(float))
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate ATR for stop loss (using 20 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA13 (13), ATR (20), and Supertrend (10)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_13[i]) or np.isnan(atr[i]) or 
            np.isnan(supertrend_dir_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema13_val = ema_13[i]
        atr_value = atr[i]
        supertrend_dir = supertrend_dir_12h_aligned[i]
        bull_pwr = bull_power[i]
        bear_pwr = bear_power[i]
        
        # Volume spike: current volume > 1.8 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.8 * vol_ma_20
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or regime/volume change
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.5*ATR from highest since entry
                if curr_close < highest_since_entry - 2.5 * atr_value:
                    exit_signal = True
                # Exit if Supertrend turns bearish OR volume dies AND weak bull power
                elif supertrend_dir <= 0 or (not volume_spike and bull_pwr < 0):
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.5 * atr_value:
                    exit_signal = True
                # Exit if Supertrend turns bullish OR volume dies AND weak bear power
                elif supertrend_dir >= 0 or (not volume_spike and bear_pwr > 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Elder Ray + Supertrend regime + volume
        if position == 0:
            # Long: Supertrend bullish AND bull power positive AND volume spike
            long_condition = (supertrend_dir > 0) and (bull_pwr > 0) and volume_spike
            # Short: Supertrend bearish AND bear power negative AND volume spike
            short_condition = (supertrend_dir < 0) and (bear_pwr < 0) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Supertrend12h_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
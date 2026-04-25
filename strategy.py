#!/usr/bin/env python3
"""
1h EMA Crossover with 4h Supertrend Trend Filter and Volume Confirmation
Hypothesis: In strong trends identified by 4h Supertrend, 1h EMA(9/21) crossovers with volume confirmation capture momentum moves. Works in bull/bear via trend filter. Discrete sizing (0.20) limits fee drag (~60-120 trades over 4 years). Session filter (08-20 UTC) reduces noise.
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
    
    # Get 4h data for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h ATR(10) for Supertrend
    tr1_4h = df_4h['high'].values[1:] - df_4h['low'].values[1:]
    tr2_4h = np.abs(df_4h['high'].values[1:] - df_4h['close'].values[:-1])
    tr3_4h = np.abs(df_4h['low'].values[1:] - df_4h['close'].values[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 4h Supertrend
    hl2_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    upper_4h = hl2_4h + 3.0 * atr_4h
    lower_4h = hl2_4h - 3.0 * atr_4h
    
    supertrend_4h = np.full_like(hl2_4h, np.nan, dtype=float)
    direction_4h = np.full_like(hl2_4h, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(hl2_4h)):
        if np.isnan(supertrend_4h[i-1]):
            supertrend_4h[i] = lower_4h[i]
            direction_4h[i] = 1
        else:
            if close_4h := df_4h['close'].values[i] <= supertrend_4h[i-1]:
                upper_4h[i] = min(upper_4h[i], upper_4h[i-1])
            else:
                upper_4h[i] = max(upper_4h[i], upper_4h[i-1])
                
            if close_4h >= supertrend_4h[i-1]:
                lower_4h[i] = max(lower_4h[i], lower_4h[i-1])
            else:
                lower_4h[i] = min(lower_4h[i], lower_4h[i-1])
            
            if direction_4h[i-1] == -1 and close_4h > upper_4h[i]:
                direction_4h[i] = 1
                supertrend_4h[i] = lower_4h[i]
            elif direction_4h[i-1] == 1 and close_4h < lower_4h[i]:
                direction_4h[i] = -1
                supertrend_4h[i] = upper_4h[i]
            else:
                direction_4h[i] = direction_4h[i-1]
                supertrend_4h[i] = supertrend_4h[i-1] if direction_4h[i] == 1 else upper_4h[i]
    
    # Align Supertrend direction to 1h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Calculate 1h EMA(9) and EMA(21)
    close_s = pd.Series(close)
    ema_9 = close_s.ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1h ATR(14) for stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA21 (21) and ATR (14)
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or np.isnan(atr[i]) or 
            np.isnan(supertrend_dir_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_fast = ema_9[i]
        ema_slow = ema_21[i]
        atr_value = atr[i]
        trend = supertrend_dir_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # EMA crossover signals
        bullish_cross = ema_fast > ema_slow
        bearish_cross = ema_fast < ema_slow
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse signal
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 2.5*ATR from highest since entry
                if curr_close < highest_since_entry - 2.5 * atr_value:
                    exit_signal = True
                # Reverse crossover or trend change
                elif bearish_cross or trend == -1:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                if curr_close > lowest_since_entry + 2.5 * atr_value:
                    exit_signal = True
                # Reverse crossover or trend change
                elif bullish_cross or trend == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: EMA crossover + trend alignment + volume + session
        if position == 0:
            # Long: bullish crossover AND uptrend AND volume spike
            long_condition = bullish_cross and (trend == 1) and volume_spike
            # Short: bearish crossover AND downtrend AND volume spike
            short_condition = bearish_cross and (trend == -1) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.20
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
    
    return signals

name = "1h_EMACrossover_4hSupertrend_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0
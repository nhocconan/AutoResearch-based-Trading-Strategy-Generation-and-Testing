#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + 1w Supertrend(10,3) + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves.
Filtered by weekly Supertrend to ensure alignment with major trend and avoid counter-trend whipsaws.
Volume confirmation reduces false breakouts. Discrete sizing (0.25) targets ~50-150 trades over 4 years.
Works in bull/bear via Supertrend direction filter. Uses proper MTF data loading to avoid look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Supertrend on weekly data
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = df_1w['high'][1:].values - df_1w['low'][1:].values
    tr2 = np.abs(df_1w['high'][1:].values - df_1w['close'][:-1].values)
    tr3 = np.abs(df_1w['low'][1:].values - df_1w['close'][:-1].values)
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (df_1w['high'].values + df_1w['low'].values) / 2
    upper_band = hl_avg + multiplier * atr
    lower_band = hl_avg - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1w := df_1w['close'].values)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = upper_band[i] if direction[i] == 1 else lower_band[i]
    
    # Align Supertrend and direction to 12h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donchian_period-1, n):
        upper_channel[i] = np.max(high[i-donchian_period+1:i+1])
        lower_channel[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Calculate 12h ATR for volume spike and stop loss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.5 * 20-period average (strict)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    for i in range(0, 19):
        vol_ma_20[i] = np.mean(volume[:i+1])
    volume_spike = volume > 2.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian (20), ATR (14), and weekly data
    start_idx = max(donchian_period-1, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_donchian = upper_channel[i]
        lower_donchian = lower_channel[i]
        st_direction = direction_aligned[i]
        atr_value = atr_12h[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        bullish_breakout = curr_close > upper_donchian
        bearish_breakout = curr_close < lower_donchian
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.0*ATR from highest since entry
                if curr_close < highest_since_entry - 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close < lower_donchian or st_direction == -1:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close > upper_donchian or st_direction == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Donchian breakout + weekly Supertrend alignment + volume
        if position == 0:
            # Long: break above upper Donchian AND weekly uptrend AND volume spike
            long_condition = bullish_breakout and (st_direction == 1) and vol_spike
            # Short: break below lower Donchian AND weekly downtrend AND volume spike
            short_condition = bearish_breakout and (st_direction == -1) and vol_spike
            
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

name = "12h_Donchian20_Breakout_1wSupertrend_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
6h Donchian(20) breakout + weekly pivot direction + volume confirmation
Hypothesis: 6h Donchian(20) breakouts capture intermediate-term momentum. 
Weekly pivot direction (based on prior weekly Camarilla H3/L3) filters for institutional bias. 
Volume spike confirms participation. Works in bull/bear: in bull, longs with weekly bias up; 
in bear, shorts with weekly bias down. Discrete sizing (0.25) limits fee drag.
Target: 12-37 trades/year (50-150 over 4 years).
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
    
    # Get weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (H3, L3) from prior weekly bar
    camarilla_h3_w = df_1w['close'] + 1.1 * (df_1w['high'] - df_1w['low']) / 4
    camarilla_l3_w = df_1w['close'] - 1.1 * (df_1w['high'] - df_1w['low']) / 4
    camarilla_h3_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_w.values)
    camarilla_l3_w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_w.values)
    
    # Weekly pivot direction: bullish if weekly close > midpoint of H3/L3, bearish if below
    weekly_midpoint = (camarilla_h3_w + camarilla_l3_w) / 2
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint.values)
    weekly_bullish = df_1w['close'] > weekly_midpoint  # True for bullish weekly bias
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.values.astype(float))
    
    # Calculate Donchian(20) channels from price
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for volume average smoothing (20 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(camarilla_h3_w_aligned[i]) or \
           np.isnan(camarilla_l3_w_aligned[i]) or np.isnan(weekly_bullish_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        highest_20_val = highest_20[i]
        lowest_20_val = lowest_20[i]
        weekly_bias_bullish = bool(weekly_bullish_aligned[i])
        atr_value = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Donchian breakout conditions
        bullish_breakout = curr_close > highest_20_val
        bearish_breakout = curr_close < lowest_20_val
        
        # Exit conditions: opposite Donchian breakout or loss of weekly bias
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish Donchian breakout or weekly bias turns bearish
                if bearish_breakout or not weekly_bias_bullish:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish Donchian breakout or weekly bias turns bullish
                if bullish_breakout or weekly_bias_bullish:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Donchian breakout + weekly bias alignment + volume spike
        if position == 0:
            # Long: bullish Donchian breakout AND weekly bullish bias AND volume spike
            long_condition = bullish_breakout and weekly_bias_bullish and volume_spike
            # Short: bearish Donchian breakout AND weekly bearish bias AND volume spike
            short_condition = bearish_breakout and (not weekly_bias_bullish) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0
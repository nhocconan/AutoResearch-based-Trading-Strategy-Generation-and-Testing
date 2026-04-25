#!/usr/bin/env python3
"""
6h Camarilla R4/S4 Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Weekly Camarilla pivots (R4/S4) from 1w represent major institutional support/resistance. 
Breakouts above R4 or below S4 with daily EMA50 trend filter and volume spike (>2x avg) capture strong momentum. 
Works in bull (buy R4 breakouts in uptrend) and bear (sell S4 breakdowns in downtrend) via symmetric logic. 
Target: 12-37 trades/year on 6h to avoid fee drag.
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
    
    # Get weekly data for Camarilla pivots (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (R4, S4) from weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_r4 = np.full(len(df_1w), np.nan)
    camarilla_s4 = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i == 0:  # First bar has no previous week
            continue
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        prev_close = close_1w[i-1]
        diff = prev_high - prev_low
        camarilla_r4[i] = prev_close + diff * 1.1 / 2  # R4 level
        camarilla_s4[i] = prev_close - diff * 1.1 / 2  # S4 level
    
    # Align weekly Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stop management
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA50, ATR, volume MA
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        uptrend = curr_close > ema_50_val
        downtrend = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals at weekly Camarilla levels
            # Long: price breaks above R4 with volume confirmation in uptrend
            long_breakout = (curr_close > r4_val) and volume_confirm and uptrend
            # Short: price breaks below S4 with volume confirmation in downtrend
            short_breakout = (curr_close < s4_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below S4 OR 2.5*ATR trailing stop OR EMA50 trend turns down
            if curr_close < s4_val or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above R4 OR 2.5*ATR trailing stop OR EMA50 trend turns up
            if curr_close > r4_val or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1wPivot_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0
#!/usr/bin/env python3
"""
1d Camarilla H3/L3 Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) from 1d represent strong intraday support/resistance. 
Breakouts above H3 or below L3 with 1w EMA34 trend filter and volume spike capture institutional momentum. 
Works in bull (buy H3 breakouts in uptrend) and bear (sell L3 breakdowns in downtrend) via symmetric logic. 
Timeframe: 1d (primary), HTF: 1w for trend filter. Target 7-25 trades/year to avoid fee drag.
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
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels (H3, L3) from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:  # First bar has no previous day
            continue
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d_arr[i-1]
        diff = prev_high - prev_low
        camarilla_h3[i] = prev_close + diff * 1.1 / 2
        camarilla_l3[i] = prev_close - diff * 1.1 / 2
    
    # Align Camarilla levels to 1d timeframe (no extra delay needed for pivot levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
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
    
    # Start index: need enough for EMA34, ATR, volume MA
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
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
        ema_34_val = ema_34_1w_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        h3_val = camarilla_h3_aligned[i]
        l3_val = camarilla_l3_aligned[i]
        
        # Trend filter: price relative to 1w EMA34 (stricter: require 0.5% deviation)
        uptrend = curr_close > ema_34_val * 1.005
        downtrend = curr_close < ema_34_val * 0.995
        
        # Volume confirmation: current volume > 2.5 * 20-period average (tighter)
        volume_confirm = curr_volume > 2.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla levels
            # Long: price breaks above H3 with volume confirmation in uptrend
            long_breakout = (curr_close > h3_val) and volume_confirm and uptrend
            # Short: price breaks below L3 with volume confirmation in downtrend
            short_breakout = (curr_close < l3_val) and volume_confirm and downtrend
            
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
            # Exit conditions: price closes below L3 OR 2.5*ATR trailing stop OR EMA34 trend turns down
            if curr_close < l3_val or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_34_val * 0.995:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above H3 OR 2.5*ATR trailing stop OR EMA34 trend turns up
            if curr_close > h3_val or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_34_val * 1.005:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0
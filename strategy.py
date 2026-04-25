#!/usr/bin/env python3
"""
12h Camarilla H3L3 Breakout + 1d EMA34 Trend + Volume Spike + ATR Trailing Stop
Hypothesis: Camarilla H3/L3 levels act as key support/resistance on 1d charts. Breakouts with volume confirmation in the direction of 1d EMA34 trend capture sustained moves. ATR trailing stop manages risk. Designed for 12h timeframe to limit trades (target: 12-37/year) and avoid fee drag while working in both bull and bear markets via trend filter.
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
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = np.full(len(df_1d), np.nan)
    camarilla_l3 = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        rng = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + rng * 1.1 / 4
        camarilla_l3[i] = close_1d[i] - rng * 1.1 / 4
        camarilla_h4[i] = close_1d[i] + rng * 1.1 / 2
        camarilla_l4[i] = close_1d[i] - rng * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (no extra delay needed for pivot points)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get 1d data for EMA34 trend filter (call ONCE before loop)
    close_1d_series = pd.Series(df_1d['close'])
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stop and sizing
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
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(h4_12h[i]) or 
            np.isnan(l4_12h[i]) or np.isnan(ema_34_12h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
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
        h3_val = h3_12h[i]
        l3_val = l3_12h[i]
        h4_val = h4_12h[i]
        l4_val = l4_12h[i]
        ema_34_val = ema_34_12h[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla H3/L3 levels
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
            # Exit conditions: price closes below L3 (reversal) OR 2.5*ATR trailing stop OR EMA34 trend turns down
            if curr_close < l3_val or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above H3 (reversal) OR 2.5*ATR trailing stop OR EMA34 trend turns up
            if curr_close > h3_val or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0
#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike + ATR Trail
Hypothesis: Camarilla H3/L3 levels act as strong support/resistance; breakouts with volume confirmation and 1d EMA34 trend filter capture momentum moves in both bull and bear markets. ATR-based trailing stop manages risk. Designed for 12h timeframe to limit trade frequency (target: 12-37 trades/year) and avoid fee drag, while working across BTC/ETH/SOL via symmetric long/short logic.
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic sizing and stop
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
    
    # Get 12h data for Camarilla levels (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (H3, L3) from previous 12h bar
    high_12h = pd.Series(df_12h['high'])
    low_12h = pd.Series(df_12h['low'])
    close_12h = pd.Series(df_12h['close'])
    
    # Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_high = close_12h + (1.1 * (high_12h - low_12h) / 6)
    camarilla_low = close_12h - (1.1 * (high_12h - low_12h) / 6)
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high.values)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for EMA34, ATR, volume MA, Camarilla
    start_idx = max(34, 14, 20, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i])):
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
        ema_34_val = ema_34_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        camarilla_high = camarilla_high_aligned[i]
        camarilla_low = camarilla_low_aligned[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla levels
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_high) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_low) and volume_confirm and downtrend
            
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
            # Exit conditions: price closes below Camarilla L3 OR 2*ATR trailing stop OR EMA34 trend turns down
            if curr_close < camarilla_low or curr_close < (highest_since_entry - 2.0 * atr_val) or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above Camarilla H3 OR 2*ATR trailing stop OR EMA34 trend turns up
            if curr_close > camarilla_high or curr_close > (lowest_since_entry + 2.0 * atr_val) or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_ATRTrail"
timeframe = "12h"
leverage = 1.0
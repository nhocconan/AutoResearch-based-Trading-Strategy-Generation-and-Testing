#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 1d EMA34 Trend + Volume Spike (Refined)
Hypothesis: Camarilla pivot levels (R3/S3) from 1d represent strong support/resistance. 
Breakouts above R3 or below S3 with 1d EMA34 trend filter and volume spike capture institutional momentum. 
Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend) via symmetric logic. 
Refined to reduce trade frequency: tighter volume confirmation (>2.5x avg) and stricter trend filter (price > EMA34 by 0.5%).
Target 20-30 trades/year on 4h to avoid fee drag.
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
    
    # Get 1d data for Camarilla pivots and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (R3, S3) from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:  # First bar has no previous day
            continue
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d_arr[i-1]
        diff = prev_high - prev_low
        camarilla_r3[i] = prev_close + diff * 1.1 / 4
        camarilla_s3[i] = prev_close - diff * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed for pivot levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
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
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
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
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        
        # Trend filter: price relative to 1d EMA34 (stricter: require 0.5% deviation)
        uptrend = curr_close > ema_34_val * 1.005
        downtrend = curr_close < ema_34_val * 0.995
        
        # Volume confirmation: current volume > 2.5 * 20-period average (tighter)
        volume_confirm = curr_volume > 2.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals at Camarilla levels
            # Long: price breaks above R3 with volume confirmation in uptrend
            long_breakout = (curr_close > r3_val) and volume_confirm and uptrend
            # Short: price breaks below S3 with volume confirmation in downtrend
            short_breakout = (curr_close < s3_val) and volume_confirm and downtrend
            
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
            # Exit conditions: price closes below S3 OR 2.5*ATR trailing stop OR EMA34 trend turns down
            if curr_close < s3_val or curr_close < (highest_since_entry - 2.5 * atr_val) or curr_close < ema_34_val * 0.995:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above R3 OR 2.5*ATR trailing stop OR EMA34 trend turns up
            if curr_close > r3_val or curr_close > (lowest_since_entry + 2.5 * atr_val) or curr_close > ema_34_val * 1.005:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_Refined"
timeframe = "4h"
leverage = 1.0
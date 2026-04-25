#!/usr/bin/env python3
"""
4h Camarilla H3/L3 Breakout + 12h EMA34 Trend + Volume Spike Confirmation
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance. 
Breakouts above H3 or below L3 with volume confirmation and aligned 12h EMA34 trend 
capture strong momentum moves while avoiding false breakouts in choppy markets.
Designed for BTC/ETH with 75-200 total trades over 4 years to balance opportunity 
and fee drag. Works in bull markets (trend continuation) and bear markets (trend 
continuation down) by using 12h EMA34 as trend filter.
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
    
    # Get daily data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # H4 = close + 1.5*(high-low)/2
    # L4 = close - 1.5*(high-low)/2
    # We use H3/L3 as entry levels, H4/L4 as stop loss
    
    # Calculate for each bar using previous day's data
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    # Need to align daily data to 4h bars
    for i in range(n):
        # Find the most recent completed daily bar
        # We'll use a simple approach: for each 4h bar, use the prior day's OHLC
        # This is handled by align_htf_to_ltf with proper delay
        
        # Get prior day's OHLC (shifted by 1 to avoid look-ahead)
        if i >= 0:  # We'll fill this in the loop after alignment
            pass
    
    # Proper MTF approach: calculate daily Camarilla, then align
    # Calculate daily Camarilla levels
    prev_high_1d = np.roll(high_1d, 1)  # previous day's high
    prev_low_1d = np.roll(low_1d, 1)    # previous day's low
    prev_close_1d = np.roll(close_1d, 1) # previous day's close
    
    # First day has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate Camarilla levels for each day
    camarilla_h3_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_h4_1d = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l4_1d = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d) / 2
    
    # Align to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # Get 12h data for EMA34 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need 34 for EMA34
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 20-period volume MA for volume spike confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34, volume MA, and Camarilla levels
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        camarilla_h3 = camarilla_h3_aligned[i]
        camarilla_l3 = camarilla_l3_aligned[i]
        camarilla_h4 = camarilla_h4_aligned[i]
        camarilla_l4 = camarilla_l4_aligned[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 12h EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_h3) and volume_confirm and uptrend
            # Short: price breaks below L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_l3) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below L3 OR EMA34 trend turns down
            if curr_close < camarilla_l3 or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above H3 OR EMA34 trend turns up
            if curr_close > camarilla_h3 or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0
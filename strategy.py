#!/usr/bin/env python3
"""
12h Camarilla H3/L3 Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as significant intraday resistance/support on 1d data.
Breakouts above H3 or below L3 with volume confirmation and aligned 1w EMA34 trend capture
swing moves in both bull and bear markets. The 1w EMA34 filter ensures we trade with the 
higher timeframe trend, reducing false breakouts. Designed for 12h timeframe with 50-150 
total trades over 4 years to balance opportunity and fee drag.
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
    
    # Get daily data for Camarilla calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need sufficient daily data
        return np.zeros(n)
    
    # Get weekly data for EMA34 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need sufficient weekly data
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H3 = Close + 1.1*(High - Low)/2
    # L3 = Close - 1.1*(High - Low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (1 bar delay for completed daily bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period volume MA for volume spike confirmation (12h)
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
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_val
        downtrend = curr_close < ema_34_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Camarilla H3 with volume confirmation in uptrend
            long_breakout = (curr_close > camarilla_h3_val) and volume_confirm and uptrend
            # Short: price breaks below Camarilla L3 with volume confirmation in downtrend
            short_breakout = (curr_close < camarilla_l3_val) and volume_confirm and downtrend
            
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
            # Exit long: price closes below Camarilla H3 OR EMA34 trend turns down
            if curr_close < camarilla_h3_val or curr_close < ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla L3 OR EMA34 trend turns up
            if curr_close > camarilla_l3_val or curr_close > ema_34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
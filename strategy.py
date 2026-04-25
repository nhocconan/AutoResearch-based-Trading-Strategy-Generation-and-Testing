#!/usr/bin/env python3
"""
1h 4h/1d Camarilla H3L3 Breakout with Volume Spike and Session Filter
Hypothesis: Camarilla H3/L3 levels act as intraday support/resistance on 4h.
Breakouts above H3 (bullish) or below L3 (bearish) with volume confirmation
and aligned with 1d EMA34 trend capture momentum moves. Session filter (08-20 UTC)
reduces noise. Discrete sizing (0.20) targets ~15-35 trades/year to avoid fee drag.
Uses 1h only for entry timing, 4h/1d for signal direction.
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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla H3 and L3 (based on previous 4h bar)
    # H3 = close + 1.1*(high - low)/2
    # L3 = close - 1.1*(high - low)/2
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Align to 1h (completed 4h bar only)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    open_time = prices['open_time']
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for alignment
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h3_level = h3_4h_aligned[i]
        l3_level = l3_4h_aligned[i]
        ema_trend = ema_34_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > h3_level
        bearish_breakout = curr_close < l3_level
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below L3 or rejects trend
                if curr_close < l3_level or curr_close < ema_trend:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above H3 or rejects trend
                if curr_close > h3_level or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume + session
        if position == 0:
            # Long: break above H3 AND price above 1d EMA34 AND volume spike
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below L3 AND price below 1d EMA34 AND volume spike
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4h_1dEMA34_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0
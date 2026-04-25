#!/usr/bin/env python3
"""
1d Camarilla Pivot H3/L3 Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance. 
Breakouts above H3 or below L3 with volume confirmation capture institutional moves. 
1w EMA34 filter ensures alignment with weekly trend (longs in weekly uptrend, shorts in downtrend).
Works in both bull and bear markets by following the higher timeframe direction.
Target: 20-50 trades/year (80-200 over 4 years) with discrete sizing 0.30 to minimize fee drag.
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index 1 (need at least 1 day of data)
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need previous day's OHLC for Camarilla calculation (index i-1)
        if i-1 < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        prev_high = df_1d['high'].values[i-1] if i-1 < len(df_1d) else df_1d['high'].values[-1]
        prev_low = df_1d['low'].values[i-1] if i-1 < len(df_1d) else df_1d['low'].values[-1]
        prev_close = df_1d['close'].values[i-1] if i-1 < len(df_1d) else df_1d['close'].values[-1]
        
        # Calculate Camarilla pivot levels for today (based on previous day)
        # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_h3 = prev_close + 1.1 * range_val / 2
        camarilla_l3 = prev_close - 1.1 * range_val / 2
        
        curr_close = close[i]
        curr_volume = volume[i]
        weekly_trend = ema_34_1w_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average (using 1d data approximated)
        # For 1d timeframe, use last 20 days volume average
        vol_lookback = min(20, i)
        if vol_lookback > 0:
            vol_ma_20 = np.mean(volume[i-vol_lookback:i])
        else:
            vol_ma_20 = np.mean(volume[:i]) if i > 0 else volume[i]
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions
        bullish_breakout = curr_close > camarilla_h3
        bearish_breakout = curr_close < camarilla_l3
        
        # Exit conditions: reverse breakout or trend rejection
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit on bearish breakout below L3 or trend rejection
                if bearish_breakout or curr_close < weekly_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Exit on bullish breakout above H3 or trend rejection
                if bullish_breakout or curr_close > weekly_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Camarilla breakout + weekly trend alignment + volume spike
        if position == 0:
            # Long: break above H3 AND price above weekly EMA34
            long_condition = bullish_breakout and (curr_close > weekly_trend) and volume_spike
            # Short: break below L3 AND price below weekly EMA34
            short_condition = bearish_breakout and (curr_close < weekly_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            signals[i] = 0.30
        elif position == -1:
            signals[i] = -0.30
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0
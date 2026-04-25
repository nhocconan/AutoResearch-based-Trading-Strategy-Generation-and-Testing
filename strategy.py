#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 1d Supertrend Filter and Volume Spike
Hypothesis: Donchian(20) breakouts on 4h capture strong momentum. 1d Supertrend filter ensures we only trade in the direction of the daily trend (long in uptrend, short in downtrend). Volume spike confirms institutional participation. Works in bull markets via long breakouts and in bear markets via short breakdowns. ATR-based trailing stop manages risk. Tight entry conditions target 75-200 total trades over 4 years to avoid fee drag.
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
    
    # Get 1d data for Supertrend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend (ATR=10, mult=3.0)
    # True Range
    tr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
    # ATR(10)
    atr_1d = np.zeros(len(df_1d))
    for i in range(10, len(df_1d)):
        atr_1d[i] = np.mean(tr_1d[i-9:i+1])
    # Basic Upper and Lower Bands
    basic_ub = (df_1d['high'].values + df_1d['low'].values) / 2 + 3.0 * atr_1d
    basic_lb = (df_1d['high'].values + df_1d['low'].values) / 2 - 3.0 * atr_1d
    # Final Upper and Lower Bands
    final_ub = np.copy(basic_ub)
    final_lb = np.copy(basic_lb)
    for i in range(1, len(df_1d)):
        if basic_ub[i] < final_ub[i-1] or df_1d['close'].iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
        if basic_lb[i] > final_lb[i-1] or df_1d['close'].iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    # Supertrend
    supertrend_1d = np.zeros(len(df_1d))
    supertrend_1d[0] = final_ub[0]
    for i in range(1, len(df_1d)):
        if supertrend_1d[i-1] == final_ub[i-1]:
            supertrend_1d[i] = final_lb[i] if df_1d['close'].iloc[i] <= final_ub[i] else final_ub[i]
        else:
            supertrend_1d[i] = final_ub[i] if df_1d['close'].iloc[i] >= final_lb[i] else final_lb[i]
    # Trend: 1 = uptrend (price above Supertrend), -1 = downtrend (price below Supertrend)
    trend_1d = np.where(close_1d := df_1d['close'].values > supertrend_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 4h Donchian(20) channels
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume confirmation (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (4h)
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Donchian, volume MA, ATR, and 1d Supertrend to propagate
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(trend_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        trend = trend_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average (strict filter)
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long breakout: close above Donchian high with volume confirmation and 1d uptrend
            long_breakout = (curr_close > donch_high_val) and volume_confirm and (trend == 1)
            # Short breakdown: close below Donchian low with volume confirmation and 1d downtrend
            short_breakout = (curr_close < donch_low_val) and volume_confirm and (trend == -1)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 1.5 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 1.5 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 1.5*ATR
            atr_stop = max(atr_stop, curr_high - 1.5 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 1.5*ATR
            atr_stop = min(atr_stop, curr_low + 1.5 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dSupertrend_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0
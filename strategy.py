#!/usr/bin/env python3
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
    
    # Get daily data for HL2 and pivot calculations
    df_1d = get_hlf_data(prices, '1d')
    if len(df_1d) < 3:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily HL2 (average of high and low)
    hl2_1d = (high_1d + low_1d) / 2
    
    # Calculate weekly HL2 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    hl2_1w = (high_1w + low_1w) / 2
    
    # Align daily HL2 and weekly HL2 to 6h timeframe
    hl2_1d_aligned = align_htf_to_ltf(prices, df_1d, hl2_1d)
    hl2_1w_aligned = align_htf_to_ltf(prices, df_1w, hl2_1w)
    
    # Calculate 6-period RSI on close for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(6, n):
        if i == 6:
            avg_gain[i] = np.mean(gain[1:7])
            avg_loss[i] = np.mean(loss[1:7])
        else:
            avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
            avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_6 = np.full(n, np.nan)
    rsi_6[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(6, 2)
    
    for i in range(start_idx, n):
        if (np.isnan(hl2_1d_aligned[i]) or 
            np.isnan(hl2_1w_aligned[i]) or
            np.isnan(rsi_6[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Determine trend direction from weekly HL2
        weekly_uptrend = hl2_1w_aligned[i] > hl2_1w_aligned[i-1]
        weekly_downtrend = hl2_1w_aligned[i] < hl2_1w_aligned[i-1]
        
        # Price relative to daily HL2 (pivot)
        price_above_hl2 = price > hl2_1d_aligned[i]
        price_below_hl2 = price < hl2_1d_aligned[i]
        
        if position == 0:
            # Long: Price above daily HL2 + weekly uptrend + RSI > 50 (bullish momentum)
            if (price_above_hl2 and 
                weekly_uptrend and 
                rsi_6[i] > 50):
                signals[i] = 0.25
                position = 1
            # Short: Price below daily HL2 + weekly downtrend + RSI < 50 (bearish momentum)
            elif (price_below_hl2 and 
                  weekly_downtrend and 
                  rsi_6[i] < 50):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price below daily HL2 or weekly trend turns down
            if (price_below_hl2 or 
                not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above daily HL2 or weekly trend turns up
            if (price_above_hl2 or 
                not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_HL2_Pivot_WeeklyTrend_RSI6_v1"
timeframe = "6h"
leverage = 1.0
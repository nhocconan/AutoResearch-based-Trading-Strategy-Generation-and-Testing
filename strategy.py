#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrendFilter_Volume
# Strategy: Camarilla pivot breakout on 1d with 1w trend filter and volume confirmation
# Long when price breaks above R1 with 1w uptrend and volume spike
# Short when price breaks below S1 with 1w downtrend and volume spike
# Exit when price touches H4 or L4 levels
# Designed for 1d timeframe with selective entries to minimize trade frequency and capture trend continuation

name = "1d_Camarilla_R1_S1_Breakout_1wTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter using EMA(34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w ATR for volume spike threshold
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    tr1 = high_1w[1:] - low_1w[:-1]
    tr2 = np.abs(high_1w[1:] - close_1w_arr[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w_arr[:-1])
    tr = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w_arr[0]), np.abs(low_1w[0] - close_1w_arr[0])])], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate daily ATR for Camarilla levels
    tr_daily = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr_daily = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], tr_daily])
    atr_daily = pd.Series(tr_daily).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day
    camarilla_R1 = np.zeros(n)
    camarilla_S1 = np.zeros(n)
    camarilla_H4 = np.zeros(n)
    camarilla_L4 = np.zeros(n)
    
    for i in range(1, n):
        # Previous day's OHLC
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        # Camarilla calculations
        camarilla_R1[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
        camarilla_S1[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
        camarilla_H4[i] = prev_close + 1.1 * (prev_high - prev_low) / 2
        camarilla_L4[i] = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume spike detection (volume > 1.5 * 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or 
            np.isnan(camarilla_H4[i]) or np.isnan(camarilla_L4[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 with 1w uptrend and volume spike
            if (close[i] > camarilla_R1[i] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and  # 1w uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with 1w downtrend and volume spike
            elif (close[i] < camarilla_S1[i] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and  # 1w downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches H4 level
            if close[i] >= camarilla_H4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches L4 level
            if close[i] <= camarilla_L4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
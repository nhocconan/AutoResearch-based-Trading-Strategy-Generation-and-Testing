#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Uses Camarilla pivot levels from daily timeframe. Enters long when price breaks above R1 with volume confirmation and 12-hour uptrend (close > EMA50).
# Enters short when price breaks below S1 with volume confirmation and 12-hour downtrend (close < EMA50).
# Exits when price returns to the pivot point (CP) or reverses direction.
# Uses 12-hour EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    CP = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    CP_aligned = align_htf_to_ltf(prices, df_1d, CP)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(CP_aligned[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and uptrend
            if (close[i] > R1_aligned[i] and 
                volume_confirm[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation and downtrend
            elif (close[i] < S1_aligned[i] and 
                  volume_confirm[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot point or trend reverses
            if (close[i] <= CP_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot point or trend reverses
            if (close[i] >= CP_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
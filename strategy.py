#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Uses weekly trend filter (price > EMA50 weekly) to avoid counter-trend trades.
# Enters long when price breaks above daily R1 with volume confirmation and weekly uptrend.
# Enters short when price breaks below daily S1 with volume confirmation and weekly downtrend.
# Exits when price returns to daily pivot point (CP) or weekly trend reverses.
# Uses 12h timeframe to target 12-37 trades per year with position size 0.25.
# Weekly trend filter reduces whipsaws and works in both bull/bear markets.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily EMA(34) for additional trend confirmation (optional)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Calculate Camarilla levels
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    CP = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 12h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    CP_aligned = align_htf_to_ltf(prices, df_1d, CP)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 20)  # Warmup for weekly EMA, daily EMA, and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(CP_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        # Optional: daily trend confirmation
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and weekly uptrend
            if (close[i] > R1_aligned[i] and 
                volume_confirm[i] and 
                price_above_weekly_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume confirmation and weekly downtrend
            elif (close[i] < S1_aligned[i] and 
                  volume_confirm[i] and 
                  price_below_weekly_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot point or weekly trend reverses
            if (close[i] <= CP_aligned[i] or 
                price_below_weekly_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot point or weekly trend reverses
            if (close[i] >= CP_aligned[i] or 
                price_above_weekly_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
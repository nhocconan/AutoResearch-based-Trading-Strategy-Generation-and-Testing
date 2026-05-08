#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Daily Trend and Volume Spike
# - Uses weekly pivot levels (S1/S2 for long, R1/R2 for short)
# - Breakout above S1 with daily uptrend or below R1 with daily downtrend
# - Volume spike confirms breakout strength
# - Weekly pivot provides robust support/resistance in both bull/bear markets
# - Daily trend filter avoids counter-trend trades
# - Target: 15-30 trades/year to minimize fee drag on 6h timeframe

name = "6h_WeeklyPivotBreakout_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels using previous week's data
    # Pivot = (H+L+C)/3
    # S1 = 2*Pivot - H, S2 = Pivot - (H-L)
    # R1 = 2*Pivot - L, R2 = Pivot + (H-L)
    n1w = len(close_1w)
    pivot = np.full(n1w, np.nan)
    weekly_S1 = np.full(n1w, np.nan)
    weekly_S2 = np.full(n1w, np.nan)
    weekly_R1 = np.full(n1w, np.nan)
    weekly_R2 = np.full(n1w, np.nan)
    
    for i in range(1, n1w):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        P = (H + L + C) / 3.0
        pivot[i] = P
        weekly_S1[i] = 2 * P - H
        weekly_S2[i] = P - (H - L)
        weekly_R1[i] = 2 * P - L
        weekly_R2[i] = P + (H - L)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_S1_aligned = align_htf_to_ltf(prices, df_1w, weekly_S1)
    weekly_S2_aligned = align_htf_to_ltf(prices, df_1w, weekly_S2)
    weekly_R1_aligned = align_htf_to_ltf(prices, df_1w, weekly_R1)
    weekly_R2_aligned = align_htf_to_ltf(prices, df_1w, weekly_R2)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter (more responsive than EMA50)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_S1_aligned[i]) or np.isnan(weekly_S2_aligned[i]) or 
            np.isnan(weekly_R1_aligned[i]) or np.isnan(weekly_R2_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S1 (support) with daily uptrend + volume spike
            long_cond = (close[i] > weekly_S1_aligned[i] and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R1 (resistance) with daily downtrend + volume spike
            short_cond = (close[i] < weekly_R1_aligned[i] and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S2 (strong support break)
            if close[i] < weekly_S2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R2 (strong resistance break)
            if close[i] > weekly_R2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
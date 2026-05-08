#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Breakout with Daily Trend and Volume Spike
# - Uses weekly pivot levels (S1/S2 for long, R1/R2 for short) from 1w timeframe
# - Breakout above S1 with daily uptrend or below R1 with daily downtrend
# - Volume spike confirms breakout strength
# - Works in bull/bear by using daily trend filter to avoid counter-trend trades
# - Target: 12-30 trades/year to minimize fee drag on 6h timeframe
# - Weekly pivots provide stronger structural levels than daily, reducing false breaks

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
    
    # 1w data for Weekly Pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot levels using previous week's data
    # Standard pivot: P = (H + L + C) / 3
    # S1 = (2*P) - H, S2 = P - (H - L)
    # R1 = (2*P) - L, R2 = P + (H - L)
    n1w = len(close_1w)
    pivot_1w = np.full(n1w, np.nan)
    support_s1_1w = np.full(n1w, np.nan)
    support_s2_1w = np.full(n1w, np.nan)
    resistance_r1_1w = np.full(n1w, np.nan)
    resistance_r2_1w = np.full(n1w, np.nan)
    
    for i in range(1, n1w):
        H = high_1w[i-1]
        L = low_1w[i-1]
        C = close_1w[i-1]
        pivot_1w[i] = (H + L + C) / 3.0
        support_s1_1w[i] = (2 * pivot_1w[i]) - H
        support_s2_1w[i] = pivot_1w[i] - (H - L)
        resistance_r1_1w[i] = (2 * pivot_1w[i]) - L
        resistance_r2_1w[i] = pivot_1w[i] + (H - L)
    
    # Align Weekly Pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    support_s1_1w_aligned = align_htf_to_ltf(prices, df_1w, support_s1_1w)
    support_s2_1w_aligned = align_htf_to_ltf(prices, df_1w, support_s2_1w)
    resistance_r1_1w_aligned = align_htf_to_ltf(prices, df_1w, resistance_r1_1w)
    resistance_r2_1w_aligned = align_htf_to_ltf(prices, df_1w, resistance_r2_1w)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
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
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(support_s1_1w_aligned[i]) or 
            np.isnan(support_s2_1w_aligned[i]) or np.isnan(resistance_r1_1w_aligned[i]) or 
            np.isnan(resistance_r2_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above S1 (support) with daily uptrend + volume spike
            long_cond = (close[i] > support_s1_1w_aligned[i] and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below R1 (resistance) with daily downtrend + volume spike
            short_cond = (close[i] < resistance_r1_1w_aligned[i] and 
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
            if close[i] < support_s2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R2 (strong resistance break)
            if close[i] > resistance_r2_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
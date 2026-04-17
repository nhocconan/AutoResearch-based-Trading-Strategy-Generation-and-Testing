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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using previous week's data
    # Group daily data into weeks (7 days per week)
    weeks_high = []
    weeks_low = []
    weeks_close = []
    
    for i in range(0, len(high_1d), 7):
        week_high = np.max(high_1d[i:i+7]) if i+7 <= len(high_1d) else np.max(high_1d[i:])
        week_low = np.min(low_1d[i:i+7]) if i+7 <= len(low_1d) else np.min(low_1d[i:])
        week_close = close_1d[i+6] if i+6 < len(close_1d) else close_1d[-1]
        weeks_high.append(week_high)
        weeks_low.append(week_low)
        weeks_close.append(week_close)
    
    weeks_high = np.array(weeks_high)
    weeks_low = np.array(weeks_low)
    weeks_close = np.array(weeks_close)
    
    # Calculate weekly pivot points: P = (H + L + C) / 3
    weekly_pivot = (weeks_high + weeks_low + weeks_close) / 3
    # Weekly R1 = 2*P - L, S1 = 2*P - H
    weekly_r1 = 2 * weekly_pivot - weeks_low
    weekly_s1 = 2 * weekly_pivot - weeks_high
    # Weekly R2 = P + (H - L), S2 = P - (H - L)
    weekly_r2 = weekly_pivot + (weeks_high - weeks_low)
    weekly_s2 = weekly_pivot - (weeks_high - weeks_low)
    
    # Expand weekly arrays back to daily frequency
    pivot_daily = np.repeat(weekly_pivot, 7)[:len(high_1d)]
    r1_daily = np.repeat(weekly_r1, 7)[:len(high_1d)]
    s1_daily = np.repeat(weekly_s1, 7)[:len(high_1d)]
    r2_daily = np.repeat(weekly_r2, 7)[:len(high_1d)]
    s2_daily = np.repeat(weekly_s2, 7)[:len(high_1d)]
    
    # Align weekly pivot levels to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_daily)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_daily)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_daily)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_daily)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_daily)
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma20 = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need volume MA20 and ATR MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma20[i]) or 
            np.isnan(pivot_6h[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        # Volatility filter: ATR > ATR MA20 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma20[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility
            if close[i] > r1_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility
            elif close[i] < s1_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below R1 or volatility drops
            if close[i] < r1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1 or volatility drops
            if close[i] > s1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_VolVol"
timeframe = "6h"
leverage = 1.0
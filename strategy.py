#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for weekly pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for daily pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Weekly Pivot Points (from previous week) ===
    # Pivot point = (H + L + C) / 3
    # Support 1 = (2 * P) - H
    # Resistance 1 = (2 * P) - L
    # Support 2 = P - (H - L)
    # Resistance 2 = P + (H - L)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels for previous week
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = (2 * pivot_1w) - high_1w
    s1_1w = (2 * pivot_1w) - low_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h timeframe (use previous week's values)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === Daily Pivot Points (from previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for previous day
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = (2 * pivot_1d) - high_1d
    s1_1d = (2 * pivot_1d) - low_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r4_1d = high_1d + 3 * (pivot_1d - low_1d)
    s4_1d = low_1d - 3 * (high_1d - pivot_1d)
    
    # Align daily pivots to 6h timeframe (use previous day's values)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === Weekly Trend Filter (EMA21 on weekly close) ===
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # === Daily Trend Filter (EMA50 on daily close) ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume confirmation (20-period average on 6h) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema21_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_pivot = pivot_1w_aligned[i]
        weekly_r1 = r1_1w_aligned[i]
        weekly_s1 = s1_1w_aligned[i]
        weekly_r2 = r2_1w_aligned[i]
        weekly_s2 = s2_1w_aligned[i]
        daily_pivot = pivot_1d_aligned[i]
        daily_r1 = r1_1d_aligned[i]
        daily_s1 = s1_1d_aligned[i]
        daily_r2 = r2_1d_aligned[i]
        daily_s2 = s2_1d_aligned[i]
        daily_r3 = r3_1d_aligned[i]
        daily_s3 = s3_1d_aligned[i]
        daily_r4 = r4_1d_aligned[i]
        daily_s4 = s4_1d_aligned[i]
        weekly_ema21 = ema21_1w_aligned[i]
        daily_ema50 = ema50_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above weekly pivot and weekly EMA21, with daily R1/S1 bounce and volume
            if (price_close > weekly_pivot and price_close > weekly_ema21 and
                ((price_close > daily_r1 and price_close < daily_r2) or  # Between R1-R2
                 (price_close > daily_s1 and price_close < daily_s2)) and  # Between S1-S2
                vol_ratio_val > 1.3):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly pivot and weekly EMA21, with daily R1/S1 bounce and volume
            elif (price_close < weekly_pivot and price_close < weekly_ema21 and
                  ((price_close > daily_r1 and price_close < daily_r2) or  # Between R1-R2
                   (price_close > daily_s1 and price_close < daily_s2)) and  # Between S1-S2
                  vol_ratio_val > 1.3):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Exit long: price breaks below weekly S1 or weekly pivot, or volume dries up
                if (price_close < weekly_s1 or price_close < weekly_pivot or vol_ratio_val < 0.7):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price breaks above weekly R1 or weekly pivot, or volume dries up
                if (price_close > weekly_r1 or price_close > weekly_pivot or vol_ratio_val < 0.7):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Daily_Pivot_Bounce_Trend_Filter"
timeframe = "6h"
leverage = 1.0
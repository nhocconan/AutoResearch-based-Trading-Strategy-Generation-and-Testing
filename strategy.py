#!/usr/bin/env python3
"""
Hypothesis: 6h Weekly Pivot Breakout with Volume Surge and Trend Filter.
Longs when price breaks above weekly R1 with volume > 2x average and ADX(14) > 20.
Shorts when price breaks below weekly S1 with volume > 2x average and ADX(14) > 20.
Exit when price crosses back below/above weekly pivot point or 1.5x ATR stop.
Weekly pivots derived from prior week's range. Designed for 15-30 trades/year on 6h timeframe.
Uses weekly timeframe for structure and 6h for execution to reduce noise and false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for pivot points and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Pivot Points (Standard formula)
    # PP = (High + Low + Close)/3
    # R1 = 2*PP - Low
    # S1 = 2*PP - High
    range_1w = high_1w - low_1w
    pivot_point = (high_1w + low_1w + close_1w) / 3
    weekly_r1 = 2 * pivot_point - low_1w
    weekly_s1 = 2 * pivot_point - high_1w
    
    # Calculate 14-period ADX for trend filter on weekly data
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    plus_dm[1:] = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                           np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm[1:] = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                            np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align Weekly Pivot Points and ADX to 6h timeframe
    pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: volume surge > 2x 50-period average on 6h
    vol_ma_50 = pd.Series(prices['volume'].values).rolling(window=50, min_periods=50).mean().values
    vol_ratio = prices['volume'].values / vol_ma_50
    
    # ATR for stoploss (50-period) on 6h
    tr1 = prices['high'].values - prices['low'].values
    tr2 = np.abs(prices['high'].values - np.roll(prices['close'].values, 1))
    tr3 = np.abs(prices['low'].values - np.roll(prices['close'].values, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_point_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        pp = pivot_point_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        adx_val = adx_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_val = atr[i]
        
        if position == 0:
            # Enter long: break above weekly R1 with volume surge and trend
            if (price_high > r1 and 
                adx_val > 20 and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly S1 with volume surge and trend
            elif (price_low < s1 and 
                  adx_val > 20 and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: pivot point cross OR ATR-based stoploss
            exit_signal = False
            
            # Pivot point exit
            if position == 1 and price_close < pp:
                exit_signal = True
            elif position == -1 and price_close > pp:
                exit_signal = True
            
            # ATR-based stoploss (1.5x ATR from pivot point as reference)
            if position == 1:
                # For longs, stop below pivot minus 1.5x ATR
                if price_close < pp - 1.5 * atr_val:
                    exit_signal = True
            elif position == -1:
                # For shorts, stop above pivot plus 1.5x ATR
                if price_close > pp + 1.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_Volume2x_ADX20"
timeframe = "6h"
leverage = 1.0
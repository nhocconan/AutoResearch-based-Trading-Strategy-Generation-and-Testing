#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Positioning_v2
Hypothesis: Weekly pivot points (PP, R1/S1, R2/S2) on 1w timeframe act as major support/resistance zones.
Price tends to respect these levels, bouncing off R1/S1 or breaking through R2/S2 with momentum.
We enter long when price bounces above S1 with bullish weekly bias, short when rejected at R1 with bearish bias.
Position size 0.25 targets ~20-30 trades/year to minimize fee drag. Uses 1d trend filter for alignment.
Works in bull markets via R2/S2 breakouts and in bear markets via R1/S1 mean reversion.
"""

name = "6h_Weekly_Pivot_Positioning_v2"
timeframe = "6h"
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
    
    # Get weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP, R1, S1, R2, S2
    # Standard formula: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H-L), S2 = PP - (H-L)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_pp = (h_1w + l_1w + c_1w) / 3.0
    weekly_r1 = 2 * weekly_pp - l_1w
    weekly_s1 = 2 * weekly_pp - h_1w
    weekly_r2 = weekly_pp + (h_1w - l_1w)
    weekly_s2 = weekly_pp - (h_1w - l_1w)
    
    # Align weekly pivots to 6h chart (waits for weekly close)
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Get 1d trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        if position == 0:
            # LONG: Price bounces above S1 with volume and bullish 1d trend
            # OR breaks above R2 with volume (strong momentum)
            long_condition = (
                (close[i] > weekly_s1_aligned[i] and close[i-1] <= weekly_s1_aligned[i-1] and volume_filter[i] and close[i] > ema50_1d_aligned[i]) or
                (close[i] > weekly_r2_aligned[i] and close[i-1] <= weekly_r2_aligned[i-1] and volume_filter[i])
            )
            
            # SHORT: Price rejected at R1 with volume and bearish 1d trend
            # OR breaks below S2 with volume (strong momentum down)
            short_condition = (
                (close[i] < weekly_r1_aligned[i] and close[i-1] >= weekly_r1_aligned[i-1] and volume_filter[i] and close[i] < ema50_1d_aligned[i]) or
                (close[i] < weekly_s2_aligned[i] and close[i-1] >= weekly_s2_aligned[i-1] and volume_filter[i])
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                
        elif position == 1:
            # EXIT LONG: Price returns to weekly PP or breaks below S1 with volume
            if (close[i] < weekly_pp_aligned[i]) or \
               (close[i] < weekly_s1_aligned[i] and volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # EXIT SHORT: Price returns to weekly PP or breaks above R1 with volume
            if (close[i] > weekly_pp_aligned[i]) or \
               (close[i] > weekly_r1_aligned[i] and volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
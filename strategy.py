#!/usr/bin/env python3
name = "12h_1w_TwoDayBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w Two-Day Breakout: 2-week high/low (10 trading days)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    two_week_high = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    two_week_low = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    
    # Align 1w breakout levels to 12h timeframe
    two_week_high_aligned = align_htf_to_ltf(prices, df_1w, two_week_high)
    two_week_low_aligned = align_htf_to_ltf(prices, df_1w, two_week_low)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h volume spike: > 2.0x 20-period average
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume > 2.0 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(two_week_high_aligned[i]) or np.isnan(two_week_low_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 2-week high with volume spike, uptrend (price > EMA34)
            if (close[i] > two_week_high_aligned[i] and vol_spike_12h[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below 2-week low with volume spike, downtrend (price < EMA34)
            elif (close[i] < two_week_low_aligned[i] and vol_spike_12h[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below 2-week low or trend reversal (price < EMA34)
            if close[i] < two_week_low_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above 2-week high or trend reversal (price > EMA34)
            if close[i] > two_week_high_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h timeframe captures multi-day momentum. 
# 2-week breakout captures institutional accumulation/distribution. 
# Volume spike confirms institutional participation. 
# EMA34 trend filter ensures trading with the trend. 
# Works in both bull/bear: breaksout in bull, breakdowns in bear. 
# Target: 15-25 trades/year, low frequency minimizes fee drag.
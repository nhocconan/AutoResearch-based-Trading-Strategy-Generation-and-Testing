#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1wTrendFilter_v1
Hypothesis: Trade breakouts of weekly Camarilla R3/S3 levels on 12h timeframe with 1-week EMA34 trend filter. Camarilla levels provide institutional support/resistance; weekly trend ensures alignment with major market direction. Discrete sizing (0.25) minimizes fee drag. Target: 12-30 trades/year to work in both bull (breakouts) and bear (mean reversion at extremes) markets.
"""

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
    
    # Get 1w data for HTF trend and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1-week EMA34 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Camarilla levels (based on prior week's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low)
    # S3 = close - 1.25*(high-low), S4 = close - 1.5*(high-low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    range_1w = high_1w - low_1w
    
    camarilla_r3 = close_1w_arr + 1.25 * range_1w
    camarilla_s3 = close_1w_arr - 1.25 * range_1w
    
    # Align Camarilla levels to 12h timeframe (no extra delay - levels known at weekly close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34 weeks)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above EMA34)
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above weekly R3 + 1w uptrend
            long_setup = (close[i] > camarilla_r3_aligned[i]) and htf_1w_bullish
            
            # Short setup: price breaks below weekly S3 + 1w downtrend
            short_setup = (close[i] < camarilla_s3_aligned[i]) and htf_1w_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches weekly S3 (mean reversion) OR 1w trend turns bearish
            if (close[i] <= camarilla_s3_aligned[i]) or (not htf_1w_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches weekly R3 (mean reversion) OR 1w trend turns bullish
            if (close[i] >= camarilla_r3_aligned[i]) or (htf_1w_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wTrendFilter_v1"
timeframe = "12h"
leverage = 1.0
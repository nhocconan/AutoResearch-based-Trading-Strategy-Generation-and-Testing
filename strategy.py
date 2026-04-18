# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_1W_Camarilla_S1_R1_Breakout_Volume
Hypothesis: Use weekly high/low as structural bias and daily Camarilla R1/S1 for entries.
Long when price breaks above weekly high and daily R1 with volume confirmation.
Short when price breaks below weekly low and daily S1 with volume confirmation.
Filters: volume > 1.5x 20-day average, avoid extreme volatility (ATR).
Position size: 0.25. Target: 10-20 trades/year to minimize fee drag.
Works in bull/bear via structural bias and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for structural bias (weekly high/low)
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Daily Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Weekly structural bias: use previous week's high/low
    prev_week_high = np.roll(high_1w, 1)
    prev_week_low = np.roll(low_1w, 1)
    prev_week_high[0] = high_1w[0]
    prev_week_low[0] = low_1w[0]
    
    # Volatility filter: ATR(20) daily to avoid choppy markets
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, prev_week_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, prev_week_low)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # Precompute volume moving average (20-day)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(atr_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid extreme volatility (stop hunting)
        atr_ma = pd.Series(atr_20_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = (np.isnan(atr_ma[i]) or atr_20_aligned[i] < atr_ma[i] * 2)
        
        if position == 0:
            # Long: price breaks above weekly high AND daily R1 with volume confirmation
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > r1_aligned[i] and 
                vol_confirm and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly low AND daily S1 with volume confirmation
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < s1_aligned[i] and 
                  vol_confirm and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly high OR daily R1
            if close[i] < weekly_high_aligned[i] or close[i] < r1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly low OR daily S1
            if close[i] > weekly_low_aligned[i] or close[i] > s1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_S1_R1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0
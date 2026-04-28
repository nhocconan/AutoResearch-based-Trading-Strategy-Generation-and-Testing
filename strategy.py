#!/usr/bin/env python3
"""
6h_Keltner_Breakout_1dTrend_WeeklyATR_Volume
Hypothesis: Uses Keltner Channel breakouts on 6h with 1d EMA50 trend filter and weekly ATR volume confirmation.
Works in both bull and bear markets by taking breakouts in direction of 1d trend, filtered by weekly ATR-based volume spikes.
Targets ~15-25 trades/year on 6f timeframe.
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for ATR and volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Keltner Channel (20-period EMA, 2*ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    kc_upper = ema_20 + 2 * atr_14
    kc_lower = ema_20 - 2 * atr_14
    
    # Calculate weekly ATR for volume filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1w = high_1w - low_1w
    tr2w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3w = np.abs(low_1w - np.roll(close_1w, 1))
    trw = np.maximum(tr1w, np.maximum(tr2w, tr3w))
    trw[0] = tr1w[0]
    atr_1w = pd.Series(trw).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume filter: volume > 1.5x 20-period MA * (current weekly ATR / average weekly ATR)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    avg_atr_1w = np.nanmean(atr_1w_aligned[~np.isnan(atr_1w_aligned)])
    vol_threshold = 1.5 * vol_ma_20 * (atr_1w_aligned / avg_atr_1w)
    vol_filter = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Keltner Channel breakout conditions
        breakout_up = high[i] > kc_upper[i-1]  # Break above upper KC
        breakout_down = low[i] < kc_lower[i-1]  # Break below lower KC
        
        # Volume confirmation
        vol_confirm = vol_filter[i]
        
        # Entry logic: Only take breakouts in direction of 1d trend
        long_entry = breakout_up and trend_up and vol_confirm
        short_entry = breakout_down and trend_down and vol_confirm
        
        # Exit logic: Opposite Keltner breakout or trend reversal
        long_exit = breakout_down or not trend_up
        short_exit = breakout_up or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Keltner_Breakout_1dTrend_WeeklyATR_Volume"
timeframe = "6h"
leverage = 1.0
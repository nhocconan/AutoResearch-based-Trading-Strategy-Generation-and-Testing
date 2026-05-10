# US
#!/usr/bin/env python3
# 6h_TurtleSoup_Reversal_Strategy
# Hypothesis: Turtle Soup pattern (false breakout reversal) works on 6h timeframe with volume confirmation and 1d trend filter.
# Long when: price breaks below 20-period low, then reverses above it with volume > 1.5x average during 1d uptrend.
# Short when: price breaks above 20-period high, then reverses below it with volume > 1.5x average during 1d downtrend.
# Works in bull/bear by fading false breakouts and following higher timeframe trend.
# Target: 15-30 trades/year per symbol.

name = "6h_TurtleSoup_Reversal_Strategy"
timeframe = "6h"
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
    
    # 60-period lookback for Turtle Soup (20-period equivalent on 6h)
    lookback = 20
    
    # Calculate rolling highs and lows
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    roll_high = high_series.rolling(window=lookback, min_periods=lookback).max().values
    roll_low = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_ma_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    vol_spike_1d = volume_1d > (vol_ma_1d * 1.5)
    
    # Align daily trend to 6h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        vol_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        vol_spike = vol_spike_1d_aligned[i] > 0.5
        
        if position == 0:
            # Turtle Soup Long: false breakdown reversal
            # Price was at or below roll_low, now back above it
            false_breakdown = (low[i] <= roll_low[i]) and (close[i] > roll_low[i])
            # Volume confirmation on the reversal bar
            if daily_up and vol_spike and false_breakdown and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Turtle Soup Short: false breakout reversal
            # Price was at or above roll_high, now back below it
            elif daily_down and vol_spike and (high[i] >= roll_high[i]) and (close[i] < roll_high[i]) and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: price returns to breakdown level or trend changes
            if close[i] <= roll_low[i] or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: price returns to breakout level or trend changes
            if close[i] >= roll_high[i] or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
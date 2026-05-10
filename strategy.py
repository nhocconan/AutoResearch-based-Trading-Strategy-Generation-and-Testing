#!/usr/bin/env python3
# 6h_WeeklyPivot_Trend_Filter_VolumeBreakout_v1
# Hypothesis: Combines weekly pivot levels (from 1w data) as structural support/resistance with 1d EMA trend filter and volume breakout confirmation.
# Weekly pivots provide robust, multi-week support/resistance that works in both trending and ranging markets.
# Entry: Break of weekly R1/S1 with volume > 2x average and alignment with 1d EMA trend.
# Exit: Price returns to weekly pivot level or trend reversal.
# Designed for low trade frequency (15-25/year) to minimize fee drag on 6h timeframe.

name = "6h_WeeklyPivot_Trend_Filter_VolumeBreakout_v1"
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
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate ATR for volatility filter (using 6-period for responsiveness)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=6, min_periods=6).mean().values
    
    # Calculate weekly pivot points from prior week's OHLC
    # Using standard floor trader pivot: P = (H + L + C)/3
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    pivot_point = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    # Weekly R1 and S1 (primary support/resistance)
    weekly_r1 = 2 * pivot_point - prev_weekly_low
    weekly_s1 = 2 * pivot_point - prev_weekly_high
    
    # Align weekly levels to 6h timeframe
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Get 1d EMA for trend filter (34-period as proven effective)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 2x average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 6)  # Warmup for volume MA, 1d EMA, and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation and volatility filter
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        volatility_filter = atr[i] > 0  # Valid ATR
        
        if position == 0:
            # Long entry: break above weekly R1 with volume and uptrend
            if close[i] > weekly_r1_aligned[i] and volume_confirm and uptrend and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly S1 with volume and downtrend
            elif close[i] < weekly_s1_aligned[i] and volume_confirm and downtrend and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: return to weekly pivot or trend reversal
            if close[i] < pivot_point[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: return to weekly pivot or trend reversal
            if close[i] > pivot_point[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
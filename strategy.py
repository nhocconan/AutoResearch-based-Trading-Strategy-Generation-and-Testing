#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining weekly pivot levels (P1/S1/R1) with volume confirmation and trend filter (EMA34).
# Weekly P1 acts as major support/resistance; breakouts with volume indicate strong momentum.
# EMA34 filters for trend alignment to avoid false breakouts in chop.
# Designed for 6h timeframe to capture medium-term breakouts with low frequency.
# Entry: Long when close > R1 and price > EMA34 and volume spike; Short when close < S1 and price < EMA34 and volume spike.
# Exit: Opposite weekly pivot level touch (S1 for long exit, R1 for short exit) or trend reversal.
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at pivot levels.
name = "6h_WeeklyPivot_EMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Pivot levels (based on prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's OHLC for Pivot calculation
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_range = prev_week_high - prev_week_low
    
    # Weekly Pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_p = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_r1 = 2 * weekly_p - prev_week_low
    weekly_s1 = 2 * weekly_p - prev_week_high
    
    # Align to 6h timeframe (waits for prior week close)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # EMA34 trend filter
    close_series = pd.Series(close)
    ema34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(ema34[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with uptrend and volume
            if (close[i] > weekly_r1_aligned[i] and 
                close[i] > ema34[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with downtrend and volume
            elif (close[i] < weekly_s1_aligned[i] and 
                  close[i] < ema34[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price touches S1 or trend turns down
            if (close[i] < weekly_s1_aligned[i]) or (close[i] < ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price touches R1 or trend turns up
            if (close[i] > weekly_r1_aligned[i]) or (close[i] > ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
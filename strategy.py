#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly high/low breakout with weekly trend filter and volume confirmation
# Uses weekly high/low as key levels for breakout/mean reversion. Weekly EMA20 ensures trend alignment.
# Volume spike >1.5 filters false breakouts. Works in bull via weekly high breakouts, in bear via weekly low breakdowns.
# Target: 10-25 trades/year to avoid fee drag. Discrete sizing 0.25.
name = "1d_WeeklyHighLow_Breakout_WeeklyEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and weekly high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate weekly high and low from previous week
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Align to daily
    prev_week_high_aligned = align_htf_to_ltf(prices, df_1w, prev_week_high)
    prev_week_low_aligned = align_htf_to_ltf(prices, df_1w, prev_week_low)
    
    # Volume confirmation - 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(prev_week_high_aligned[i]) or np.isnan(prev_week_low_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above weekly high with trend alignment and volume confirmation
            if (close[i] > prev_week_high_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short entry: break below weekly low with trend alignment and volume confirmation
            elif (close[i] < prev_week_low_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below weekly low (mean reversion) OR trend fails
            if close[i] < prev_week_low_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above weekly high (mean reversion) OR trend fails
            if close[i] > prev_week_high_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
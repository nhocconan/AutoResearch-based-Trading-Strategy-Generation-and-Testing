#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1S1_Breakout_Trend_Filter_v4"
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
    
    # Get 1d data for current close (used in calculations)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter and weekly high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w = (close_1w > ema20_1w).astype(float)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Previous week's OHLC for weekly Camarilla calculation
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    
    # Weekly Camarilla pivot levels
    pivot_week = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    range_week = prev_week_high - prev_week_low
    r1_week = pivot_week + (range_week * 1.1 / 6)
    s1_week = pivot_week - (range_week * 1.1 / 6)
    
    # Align weekly Camarilla levels to daily timeframe
    r1_week_aligned = align_htf_to_ltf(prices, df_1w, r1_week)
    s1_week_aligned = align_htf_to_ltf(prices, df_1w, s1_week)
    
    # Volume spike detection: current volume > 2.0 * 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_week_aligned[i]) or np.isnan(s1_week_aligned[i]) or 
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1 with volume spike and weekly uptrend
            long_cond = (close[i] > r1_week_aligned[i] and vol_spike[i] and trend_1w_aligned[i] > 0.5)
            
            # Short entry: price breaks below weekly S1 with volume spike and weekly downtrend
            short_cond = (close[i] < s1_week_aligned[i] and vol_spike[i] and trend_1w_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 (reversal signal)
            if close[i] < s1_week_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above weekly R1 (reversal signal)
            if close[i] > r1_week_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla R1/S1 breakout strategy with volume spike confirmation and weekly EMA20 trend filter on 1d timeframe.
# Enters long when price breaks above weekly R1 with volume spike and weekly uptrend (close > EMA20).
# Enters short when price breaks below weekly S1 with volume spike and weekly downtrend (close < EMA20).
# Exits when price reverses back through weekly S1/R1 respectively.
# Uses discrete sizing (0.25) to minimize churn. Targets 15-30 trades/year on 1d timeframe.
# Weekly trend filter ensures we only trade with the higher timeframe trend, reducing whipsaw in sideways markets.
# Works in bull markets (trend-following breakouts) and bear markets (reversal breakouts from overextended levels).
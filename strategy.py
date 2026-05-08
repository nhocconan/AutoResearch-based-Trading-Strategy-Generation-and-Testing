#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Get weekly data once for Camarilla pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Camarilla pivot levels (based on previous week)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Calculate pivot point
    pivot_w = (high_w + low_w + close_w) / 3.0
    range_w = high_w - low_w
    
    # Resistance and Support levels
    r1_w = pivot_w + (range_w * 1.0833)
    s1_w = pivot_w - (range_w * 1.0833)
    r2_w = pivot_w + (range_w * 1.1666)
    s2_w = pivot_w - (range_w * 1.1666)
    r3_w = pivot_w + (range_w * 1.2500)
    s3_w = pivot_w - (range_w * 1.2500)
    r4_w = pivot_w + (range_w * 1.5000)
    s4_w = pivot_w - (range_w * 1.5000)
    
    # Align weekly levels to daily timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Weekly trend: EMA34 on weekly close
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_w_aligned = align_htf_to_ltf(prices, df_1w, ema34_w)
    
    # Volume confirmation: volume > 1.5x 20-day average (using daily data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or np.isnan(ema34_w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and above weekly EMA34
            long_cond = (close[i] > r1_w_aligned[i]) and (volume[i] > vol_ma20[i] * 1.5) and (close[i] > ema34_w_aligned[i])
            
            # Short entry: price breaks below S1 with volume confirmation and below weekly EMA34
            short_cond = (close[i] < s1_w_aligned[i]) and (volume[i] > vol_ma20[i] * 1.5) and (close[i] < ema34_w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 or weekly trend turns bearish
            if (close[i] < s1_w_aligned[i]) or (close[i] < ema34_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 or weekly trend turns bullish
            if (close[i] > r1_w_aligned[i]) or (close[i] > ema34_w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla pivot levels (R1/S1) act as strong support/resistance in crypto markets.
# Long when price breaks above weekly R1 with volume confirmation and above weekly EMA34 trend.
# Short when price breaks below weekly S1 with volume confirmation and below weekly EMA34 trend.
# Exits when price returns to opposite S1/R1 level or weekly trend reverses.
# Weekly timeframe provides structural context, daily execution provides timely entries.
# Volume confirmation filters out false breakouts. Target: 20-60 trades over 4 years = 5-15/year.
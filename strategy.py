#!/usr/bin/env python3
# 6h_1d_1w_camarilla_pivot_volume_v1
# Strategy: 6h Camarilla pivot levels with 1d volume confirmation and 1w trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels provide high-probability reversal points. 1d volume confirms institutional participation at these levels. 1w trend filter ensures alignment with higher timeframe direction. Designed for low trade frequency to minimize fee drag in choppy markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Using typical formula: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We'll calculate these for each day and align to 6h
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    R4 = prev_day_close + 1.5 * (prev_day_high - prev_day_low)
    R3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low)
    S3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low)
    S4 = prev_day_close - 1.5 * (prev_day_high - prev_day_low)
    
    # Align Camarilla levels to 6h timeframe
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d volume confirmation: current 1d volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm_1d = vol_1d > 1.5 * vol_avg_20_1d
    vol_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d.astype(float))
    
    # 1w trend filter: EMA21 on weekly close
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required data is invalid
        if (np.isnan(R4_6h[i]) or np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(S4_6h[i]) or
            np.isnan(vol_confirm_1d_aligned[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Fade at S3/R3 levels with volume confirmation
        fade_long = (low[i] <= S3_6h[i]) and (close[i] > S3_6h[i]) and vol_confirm_1d_aligned[i] > 0.5
        fade_short = (high[i] >= R3_6h[i]) and (close[i] < R3_6h[i]) and vol_confirm_1d_aligned[i] > 0.5
        
        # Breakout continuation at S4/R4 levels with volume confirmation
        breakout_long = (high[i] > S4_6h[i]) and (close[i] > S4_6h[i]) and vol_confirm_1d_aligned[i] > 0.5
        breakdown_short = (low[i] < R4_6h[i]) and (close[i] < R4_6h[i]) and vol_confirm_1d_aligned[i] > 0.5
        
        # 1w trend filter: only take longs in uptrend, shorts in downtrend
        trend_up = close[i] > ema_21_1w_aligned[i]
        trend_down = close[i] < ema_21_1w_aligned[i]
        
        # Entry conditions
        # Long: Fade at S3 OR breakout above S4, with volume confirmation and trend alignment
        if ((fade_long or breakout_long) and trend_up and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: Fade at R3 OR breakdown below R4, with volume confirmation and trend alignment
        elif ((fade_short or breakdown_short) and trend_down and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Opposite signal or loss of volume confirmation
        elif position == 1 and (fade_short or breakdown_short or vol_confirm_1d_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (fade_long or breakout_long or vol_confirm_1d_aligned[i] <= 0.5):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
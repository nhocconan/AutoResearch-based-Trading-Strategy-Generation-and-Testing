#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Weekly_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Weekly close for Camarilla calculation
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Camarilla levels (R1, S1) from previous week's range
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w[0] = close_1w[0]
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    R1_1w = prev_close_1w + (prev_high_1w - prev_low_1w) * 1.1 / 12
    S1_1w = prev_close_1w - (prev_high_1w - prev_low_1w) * 1.1 / 12
    
    # Align weekly Camarilla levels to 12h timeframe
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    
    # Daily trend filter: EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = (close_1d > ema34_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Daily volume spike: current volume > 1.5 * 20-day average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma20d * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_1w_aligned[i]) or np.isnan(S1_1w_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above weekly R1 with volume spike and daily uptrend
            long_cond = (close[i] > R1_1w_aligned[i] and vol_spike_aligned[i] and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below weekly S1 with volume spike and daily downtrend
            short_cond = (close[i] < S1_1w_aligned[i] and vol_spike_aligned[i] and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below weekly S1 (mean reversion to support)
            if close[i] < S1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above weekly R1 (mean reversion to resistance)
            if close[i] > R1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Camarilla R1/S1 breakout on 12H with daily volume confirmation and daily trend filter.
# Weekly structure provides robust support/resistance levels that work in both bull and bear markets.
# In bull markets, price tends to continue breaking above R1; in bear markets, tends to reverse at S1.
# Daily EMA34 filter ensures alignment with intermediate-term trend to reduce counter-trend trades.
# Daily volume spike (1.5x 20-day average) confirms institutional participation in the breakout.
# Target: 15-30 trades/year to minimize fee decay while capturing significant weekly swings.
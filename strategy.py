#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with daily volume spike and weekly trend filter.
# Uses daily Camarilla levels (H4/L4) for mean reversion entries, confirmed by volume spikes
# and filtered by weekly trend direction to avoid counter-trend trades. Weekly trend avoids
# whipsaws in ranging markets. 12h timeframe targets 50-150 total trades over 4 years.
# Camarilla reversals work in both bull and bear markets as price reverts to mean.
# Volume spike ensures institutional participation. Weekly trend filter improves win rate.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h4 = prev_close + 1.1 * range_ / 2  # H4 level
    camarilla_l4 = prev_close - 1.1 * range_ / 2  # L4 level
    
    # Weekly trend: EMA crossover
    close_1w = df_1w['close'].values
    ema_fast = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    weekly_uptrend = ema_fast > ema_slow
    
    # Daily volume spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > (volume_ma_20 * 2.0)  # 2x average volume
    
    # Align all data to 12-hour timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion at Camarilla H4/L4 levels with volume spike and trend filter
        # Short near H4 in uptrend, Long near L4 in downtrend (fade the extreme)
        near_h4 = close[i] >= camarilla_h4_aligned[i] * 0.999  # Within 0.1% of H4
        near_l4 = close[i] <= camarilla_l4_aligned[i] * 1.001  # Within 0.1% of L4
        
        if position == 0:
            # Short when near H4, weekly uptrend, and volume spike
            if near_h4 and weekly_uptrend_aligned[i] >= 0.5 and volume_spike_aligned[i] >= 0.5:
                position = -1
                signals[i] = -position_size
            # Long when near L4, weekly downtrend, and volume spike
            elif near_l4 and weekly_uptrend_aligned[i] < 0.5 and volume_spike_aligned[i] >= 0.5:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long when price moves back toward midpoint or weekly trend changes
            midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] >= midpoint or weekly_uptrend_aligned[i] >= 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short when price moves back toward midpoint or weekly trend changes
            midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
            if close[i] <= midpoint or weekly_uptrend_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_Camarilla_Reversal_Volume_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0
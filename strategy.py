#!/usr/bin/env python3
# 12h_1w_camarilla_volume_trend_v1
# Strategy: 12h Camarilla pivot breakout with weekly trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaking Camarilla H4/L4 levels with volume spike and weekly trend alignment
# provides high-probability breakout trades. Weekly filter ensures we trade with higher timeframe momentum.
# Volume confirmation avoids false breakouts. Works in bull (breakouts in uptrend) and bear 
# (breakdowns in downtrend) by only trading in direction of weekly trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly close for trend filter (simple MA crossover)
    close_1w = df_1w['close'].values
    ma_fast = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ma_slow = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = ma_fast > ma_slow
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    
    # Daily data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar using previous day's OHLC
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shift by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first bar uses same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla H4 and L4 levels
    camarilla_h4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current values
        price_now = close[i]
        vol_spike_now = vol_spike[i]
        
        # Breakout conditions
        breakout_long = price_now > camarilla_h4_aligned[i]
        breakout_short = price_now < camarilla_l4_aligned[i]
        
        # Entry logic: breakout with volume spike and weekly trend alignment
        if breakout_long and vol_spike_now and weekly_uptrend_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and vol_spike_now and not weekly_uptrend_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout or loss of weekly trend
        elif position == 1 and (breakout_short or not weekly_uptrend_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_long or weekly_uptrend_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
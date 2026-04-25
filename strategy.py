#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter and choppiness regime.
Long when price breaks above R1 + 1d uptrend + chop < 61.8 (trending market).
Short when price breaks below S1 + 1d downtrend + chop < 61.8.
Exit on opposite level touch or trend/chop regime change.
Position size: 0.25 to balance return and drawdown.
Target: 15-30 trades/year to stay well under 200-trade 12h hard max.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 bars for EMA34 and chop
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d choppiness index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_1d = 100 * np.log10(sum_atr_14 / chop_denom * np.sqrt(14))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1_1d = prev_close_1d + 1.1 * camarilla_range / 12
    s1_1d = prev_close_1d - 1.1 * camarilla_range / 12
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34) and chop (14) and Camarilla (1)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend and regime
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        chop_value = chop_1d_aligned[i]
        is_trending = chop_value < 61.8  # trending market
        
        if position == 0:
            # Long setup: price breaks above R1 + 1d uptrend + trending market
            long_setup = (close[i] > r1_1d_aligned[i]) and htf_1d_bullish and is_trending
            
            # Short setup: price breaks below S1 + 1d downtrend + trending market
            short_setup = (close[i] < s1_1d_aligned[i]) and htf_1d_bearish and is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches S1 (stop) OR 1d trend turns bearish OR market becomes ranging
            if (close[i] <= s1_1d_aligned[i]) or (not htf_1d_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R1 (stop) OR 1d trend turns bullish OR market becomes ranging
            if (close[i] >= r1_1d_aligned[i]) or (htf_1d_bullish) or (not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0
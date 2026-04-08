#!/usr/bin/env python3
"""
4h_1w_1d_price_action_v2
Hypothesis: Weekly bias (from previous week close vs open) filters daily breakouts on 4h timeframe.
- Weekly bullish bias: weekly close > open → only look for long entries
- Weekly bearish bias: weekly close < open → only look for short entries  
- Daily high/low as dynamic support/resistance levels
- Enter on 4h breakout of daily levels in direction of weekly bias
- Exit on opposite daily level touch or weekly bias reversal
- Works in bull/bear via weekly filter; avoids counter-trend trades
Target: 20-40 trades/year
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_1d_price_action_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly bias: bullish if weekly close > open, bearish if close < open
    weekly_bullish = df_1w['close'].values > df_1w['open'].values
    weekly_bearish = df_1w['close'].values < df_1w['open'].values
    
    # Forward fill weekly bias to get current week's bias
    weekly_bullish_series = pd.Series(weekly_bullish)
    weekly_bearish_series = pd.Series(weekly_bearish)
    weekly_bullish_ffilled = weekly_bullish_series.ffill().values
    weekly_bearish_ffilled = weekly_bearish_series.ffill().values
    
    # Align weekly bias to 4h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish_ffilled)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish_ffilled)
    
    # Get daily data for support/resistance
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily high/low as support/resistance
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Forward fill daily levels
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    daily_high_ffilled = daily_high_series.ffill().values
    daily_low_ffilled = daily_low_series.ffill().values
    
    # Align daily levels to 4h
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high_ffilled)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low_ffilled)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches daily low (support) or weekly bias turns bearish
            if low[i] <= daily_low_aligned[i] or weekly_bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price touches daily high (resistance) or weekly bias turns bullish
            if high[i] >= daily_high_aligned[i] or weekly_bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above daily high with weekly bullish bias
            if high[i] > daily_high_aligned[i] and weekly_bullish_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below daily low with weekly bearish bias
            elif low[i] < daily_low_aligned[i] and weekly_bearish_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
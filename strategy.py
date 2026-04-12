#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_camarilla_cascade_v1
# Uses weekly pivot direction as primary trend filter, daily Camarilla for entries, and volume confirmation.
# In bull markets: weekly bullish + price breaks above daily H3/H4 = long
# In bear markets: weekly bearish + price breaks below daily L3/L4 = short
# Volume spike confirms institutional participation. Target: 15-25 trades/year per symbol.
name = "6h_1w_1d_camarilla_cascade_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly trend: bullish if weekly close > weekly open, bearish if close < open
    weekly_bullish = df_1w['close'] > df_1w['open']
    weekly_bearish = df_1w['close'] < df_1w['open']
    weekly_trend = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(int) - weekly_bearish.astype(int))
    
    # Daily Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    range_prev = high_prev - low_prev
    # Calculate H3, H4, L3, L4 levels
    daily_h3 = close_prev + range_prev * 1.1 / 4
    daily_h4 = close_prev + range_prev * 1.1 / 2
    daily_l3 = close_prev - range_prev * 1.1 / 4
    daily_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align daily levels to 6h timeframe
    h3_level = align_htf_to_ltf(prices, df_1d, daily_h3)
    h4_level = align_htf_to_ltf(prices, df_1d, daily_h4)
    l3_level = align_htf_to_ltf(prices, df_1d, daily_l3)
    l4_level = align_htf_to_ltf(prices, df_1d, daily_l4)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if levels not ready
        if (np.isnan(h3_level[i]) or np.isnan(h4_level[i]) or 
            np.isnan(l3_level[i]) or np.isnan(l4_level[i]) or
            np.isnan(weekly_trend[i])):
            signals[i] = 0.0
            continue
        
        # Skip if volume confirmation fails
        if not vol_confirm[i]:
            # Hold current position if volume fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        weekly_trend_val = weekly_trend[i]
        
        # Long signal: weekly bullish AND price breaks above H3/H4
        if weekly_trend_val > 0 and (close[i] > h3_level[i] or close[i] > h4_level[i]) and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: weekly bearish AND price breaks below L3/L4
        elif weekly_trend_val < 0 and (close[i] < l3_level[i] or close[i] < l4_level[i]) and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite weekly trend or opposite breakout
        elif (weekly_trend_val < 0 and position == 1) or \
             (weekly_trend_val > 0 and position == -1) or \
             (close[i] < l4_level[i] and position == 1) or \
             (close[i] > h4_level[i] and position == -1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals
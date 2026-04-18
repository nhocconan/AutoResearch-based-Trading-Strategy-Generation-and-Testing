#!/usr/bin/env python3
"""
1d Weekly Range Breakout with Volume Spike and Trend Filter
Hypothesis: Weekly high/low act as key support/resistance. Breakouts with volume confirmation and trend alignment capture momentum.
Works in bull/bear markets by requiring volume to avoid false breaks and trend filter to align with higher timeframe bias.
Target: 15-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for previous week's high/low (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Previous week's high and low (shifted by 1 to avoid look-ahead)
    prev_week_high = df_w['high'].shift(1).values
    prev_week_low = df_w['low'].shift(1).values
    
    # Align to 1d timeframe
    prev_week_high_aligned = align_htf_to_ltf(prices, df_w, prev_week_high)
    prev_week_low_aligned = align_htf_to_ltf(prices, df_w, prev_week_low)
    
    # Weekly trend: 34-period EMA of weekly close
    weekly_close = df_w['close'].values
    weekly_ema34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema34_aligned = align_htf_to_ltf(prices, df_w, weekly_ema34)
    
    # Volume spike: 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(prev_week_high_aligned[i]) or 
            np.isnan(prev_week_low_aligned[i]) or
            np.isnan(weekly_ema34_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        pwh = prev_week_high_aligned[i]
        pwl = prev_week_low_aligned[i]
        w_trend = weekly_ema34_aligned[i]
        
        if position == 0:
            # Long: break above weekly high with volume spike and bullish weekly trend
            if price > pwh and volume_spike[i] and price > w_trend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume spike and bearish weekly trend
            elif price < pwl and volume_spike[i] and price < w_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price returns to weekly low or trend turns bearish
            if price <= pwl or price < w_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price returns to weekly high or trend turns bullish
            if price >= pwh or price > w_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyRange_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0
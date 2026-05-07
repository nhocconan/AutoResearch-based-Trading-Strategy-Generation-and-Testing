#!/usr/bin/env python3
"""
1d_WilliamsAlligator_Vortex_Trend_Filter
Hypothesis: Daily timeframe with Williams Alligator for trend direction and Vortex for trend strength.
Williams Alligator uses smoothed medians (Jaw=13, Teeth=8, Lips=5) to filter trend.
Vortex indicator (VI+ and VI-) identifies trend direction and strength.
Long when VI+ > VI- and price > Alligator Teeth in bullish Alligator alignment.
Short when VI- > VI+ and price < Alligator Teeth in bearish Alligator alignment.
Weekly trend filter (price > weekly EMA20) ensures alignment with higher timeframe.
Designed for low trade frequency (<25/year) to minimize fee drag.
Works in bull/bear via trend filters and uses volatility-based position sizing.
"""

name = "1d_WilliamsAlligator_Vortex_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams Alligator: Smoothed medians
    # Jaw: 13-period smoothed median (SMMA of median)
    median = (high + low) / 2
    jaw_raw = pd.Series(median).rolling(window=13, min_periods=13).median().values
    jaw = pd.Series(jaw_raw).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    
    # Teeth: 8-period smoothed median
    teeth_raw = pd.Series(median).rolling(window=8, min_periods=8).median().values
    teeth = pd.Series(teeth_raw).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    
    # Lips: 5-period smoothed median
    lips_raw = pd.Series(median).rolling(window=5, min_periods=5).median().values
    lips = pd.Series(lips_raw).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    # Align Alligator components to daily
    jaw_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), jaw)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), lips)
    
    # Vortex Indicator
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Vortex movements
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Sum over 14 periods
    vi_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vi_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # VI+ and VI-
    vi_plus = np.divide(vi_plus_sum, tr_sum, out=np.zeros_like(vi_plus_sum), where=tr_sum!=0)
    vi_minus = np.divide(vi_minus_sum, tr_sum, out=np.zeros_like(vi_minus_sum), where=tr_sum!=0)
    
    # Align Vortex components
    vi_plus_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), vi_minus)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(vi_plus_aligned[i]) or
            np.isnan(vi_minus_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        if np.isnan(close_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Alligator alignment
        bullish_alligator = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        bearish_alligator = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        
        if position == 0:
            # Long: VI+ > VI-, price > Teeth, bullish Alligator, weekly uptrend
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                close[i] > teeth_aligned[i] and 
                bullish_alligator and 
                weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: VI- > VI+, price < Teeth, bearish Alligator, weekly downtrend
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  close[i] < teeth_aligned[i] and 
                  bearish_alligator and 
                  weekly_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: VI- > VI+ or price < Teeth or bearish Alligator or weekly downtrend
            if (vi_minus_aligned[i] >= vi_plus_aligned[i] or 
                close[i] <= teeth_aligned[i] or 
                not bullish_alligator or 
                not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: VI+ > VI- or price > Teeth or bullish Alligator or weekly uptrend
            if (vi_plus_aligned[i] >= vi_minus_aligned[i] or 
                close[i] >= teeth_aligned[i] or 
                not bearish_alligator or 
                not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
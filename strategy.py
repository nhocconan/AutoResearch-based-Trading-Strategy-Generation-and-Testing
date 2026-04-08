#!/usr/bin/env python3
# 6h_1w_1d_pivots_breakout_volume_v1
# Hypothesis: Weekly pivot levels with daily trend filter and volume confirmation
# Long when: Price breaks above weekly R1 + daily EMA50 rising + volume > 1.5x average
# Short when: Price breaks below weekly S1 + daily EMA50 falling + volume > 1.5x average
# Exit when price crosses back below/above weekly pivot point
# Weekly pivots provide structural support/resistance, daily EMA filters trend, volume confirms conviction
# Works in bull (breaks upward) and bear (breaks downward) markets
# Target: 15-30 trades/year with strict breakout conditions

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_pivots_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivots from previous week's data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pp_1w - low_1w
    s1_1w = 2 * pp_1w - high_1w
    
    # Align weekly pivots to 6h timeframe (wait for weekly close)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Daily EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema_1d_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below weekly pivot point
            if close[i] < pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above weekly pivot point
            if close[i] > pp_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above weekly R1, daily EMA50 rising, volume surge
            if (close[i] > r1_1w_aligned[i] and 
                ema_1d_50_aligned[i] > ema_1d_50_aligned[i-1] and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below weekly S1, daily EMA50 falling, volume surge
            elif (close[i] < s1_1w_aligned[i] and 
                  ema_1d_50_aligned[i] < ema_1d_50_aligned[i-1] and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12h_camarilla_1d_trend_volume_v4
# Hypothesis: 12h strategy using 1d Camarilla pivot levels for structure, volume confirmation for momentum, and trend filter via 1d EMA200.
# Long when: price > EMA200_1d + price touches/breaks Camarilla H3/H4 resistance + volume > 1.5x 20-period average
# Short when: price < EMA200_1d + price touches/breaks Camarilla L3/L4 support + volume > 1.5x 20-period average
# Exits when price returns to Camarilla Pivot Point (mean reversion to fair value) or trend reverses.
# Discrete sizing (±0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_trend_volume_v4"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), 
    # L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    # Pivot = (high + low + close)/3
    cam_high_1d = high_1d
    cam_low_1d = low_1d
    cam_close_1d = close_1d
    
    cam_pivot = (cam_high_1d + cam_low_1d + cam_close_1d) / 3.0
    cam_range = cam_high_1d - cam_low_1d
    cam_h4 = cam_close_1d + 1.5 * cam_range
    cam_h3 = cam_close_1d + 1.0 * cam_range
    cam_l3 = cam_close_1d - 1.0 * cam_range
    cam_l4 = cam_close_1d - 1.5 * cam_range
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    cam_pivot_aligned = align_htf_to_ltf(prices, df_1d, cam_pivot)
    cam_h3_aligned = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_h4_aligned = align_htf_to_ltf(prices, df_1d, cam_h4)
    cam_l3_aligned = align_htf_to_ltf(prices, df_1d, cam_l3)
    cam_l4_aligned = align_htf_to_ltf(prices, df_1d, cam_l4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(cam_pivot_aligned[i]) or 
            np.isnan(cam_h3_aligned[i]) or np.isnan(cam_h4_aligned[i]) or
            np.isnan(cam_l3_aligned[i]) or np.isnan(cam_l4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot point (mean reversion) or trend reverses
            if close[i] <= cam_pivot_aligned[i] or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot point (mean reversion) or trend reverses
            if close[i] >= cam_pivot_aligned[i] or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price above EMA200 + breaks above H3/H4 resistance
                if close[i] > ema200_1d_aligned[i] and close[i] > cam_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below EMA200 + breaks below L3/L4 support
                elif close[i] < ema200_1d_aligned[i] and close[i] < cam_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
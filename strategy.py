#!/usr/bin/env python3
"""
6h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 6-hour timeframe, use weekly Camarilla pivot levels with trend filter from 1-week EMA200 and volume confirmation. 
Enter long at S3 bounce in uptrend, short at R3 bounce in downtrend with volume > 1.5x average. 
Exit at opposite pivot level (S1/R1). Designed for low frequency (12-37 trades/year) to avoid fee drag while capturing 
mean reversion in strong trends. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend) by using 1-week trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    w_ema200 = pd.Series(w_close).ewm(span=200, adjust=False).mean().values
    w_ema200_aligned = align_htf_to_ltf(prices, df_1w, w_ema200)
    
    # Calculate 24-period average volume for confirmation (4 days at 6h)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if weekly EMA200 not available
        if np.isnan(w_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs weekly EMA200
        uptrend = close[i] > w_ema200_aligned[i]
        downtrend = close[i] < w_ema200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below S1
            if i >= 1:
                # Need previous week's data to calculate pivots
                wk_idx = i // 28  # Approximate: 28 six-hour bars per week
                if wk_idx >= 1 and wk_idx < len(w_high):
                    wh = w_high[wk_idx-1]
                    wl = w_low[wk_idx-1]
                    wc = w_close[wk_idx-1]
                    rng = wh - wl
                    if rng > 0:
                        s1 = wc - (1.1 * rng / 12)
                        if close[i] <= s1:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = 0.25
                    else:
                        signals[i] = 0.25
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above R1
            if i >= 1:
                wk_idx = i // 28
                if wk_idx >= 1 and wk_idx < len(w_high):
                    wh = w_high[wk_idx-1]
                    wl = w_low[wk_idx-1]
                    wc = w_close[wk_idx-1]
                    rng = wh - wl
                    if rng > 0:
                        r1 = wc + (1.1 * rng / 12)
                        if close[i] >= r1:
                            position = 0
                            signals[i] = 0.0
                        else:
                            signals[i] = -0.25
                    else:
                        signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need at least 1 week of data to calculate pivots
            wk_idx = i // 28
            if wk_idx >= 1 and wk_idx < len(w_high):
                wh = w_high[wk_idx-1]
                wl = w_low[wk_idx-1]
                wc = w_close[wk_idx-1]
                rng = wh - wl
                if rng > 0:
                    # Calculate Camarilla levels
                    s3 = wc - (1.1 * rng / 4)
                    s1 = wc - (1.1 * rng / 12)
                    r3 = wc + (1.1 * rng / 4)
                    r1 = wc + (1.1 * rng / 12)
                    
                    # Long entry: price bounces off S3 in uptrend with volume confirmation
                    long_entry = (close[i] >= s3 and close[i] <= s1) and uptrend and vol_confirm
                    # Short entry: price bounces off R3 in downtrend with volume confirmation
                    short_entry = (close[i] <= r3 and close[i] >= r1) and downtrend and vol_confirm
                    
                    if long_entry:
                        position = 1
                        signals[i] = 0.25
                    elif short_entry:
                        position = -1
                        signals[i] = -0.25
    
    return signals
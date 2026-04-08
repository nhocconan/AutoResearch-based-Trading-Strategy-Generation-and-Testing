#!/usr/bin/env python3
# 12h_Camarilla_Pivot_1w_Trend_Volume_v1
# Hypothesis: 12h Camarilla pivot reversals with 1w trend filter and volume confirmation.
# Uses 1w EMA to filter direction and Camarilla levels from 1d for entry/exit.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear by following higher timeframe trend (1w) and fading extremes at Camarilla levels.

name = "12h_Camarilla_Pivot_1w_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA trend filter (21-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1-day data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # We use H3/L3 for entries: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Align Camarilla levels to 12h timeframe (1 bar delay for previous day close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume filter: volume > 1.5x 24-period average (~12 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(21, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches Camarilla H3 or trend fails
            if close[i] >= camarilla_h3_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches Camarilla L3 or trend fails
            if close[i] <= camarilla_l3_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Fade long at L3 in uptrend: price touches L3 and closes above it
                if close[i] <= camarilla_l3_aligned[i] and close[i] > camarilla_l3_aligned[i] * 0.999 and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Fade short at H3 in downtrend: price touches H3 and closes below it
                elif close[i] >= camarilla_h3_aligned[i] and close[i] < camarilla_h3_aligned[i] * 1.001 and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
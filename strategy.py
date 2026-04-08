#!/usr/bin/env python3
# 4h_camarilla_pivot_daily_trend_volume_v2
# Hypothesis: Uses Camarilla pivot levels from daily timeframe with volume confirmation and daily trend filter.
# Goes long when price touches Camarilla L3 level in uptrend (price > daily EMA50) with volume surge.
# Goes short when price touches Camarilla H3 level in downtrend (price < daily EMA50) with volume surge.
# Designed for low trade frequency (20-50/year) to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_daily_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels (based on previous day's range)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = C + 1.1*(H-L)/2, L3 = C - 1.1*(H-L)/2
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price moves below L3 or trend change
            if close[i] < camarilla_l3_aligned[i] or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above H3 or trend change
            if close[i] > camarilla_h3_aligned[i] or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: price touches L3 in uptrend
                if daily_uptrend and close[i] <= camarilla_l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches H3 in downtrend
                elif daily_downtrend and close[i] >= camarilla_h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
# 12h_camarilla_pivot_daily_trend_volume_v1
# Hypothesis: Uses Camarilla pivot levels from 1d for entry/exit with 1w trend filter and volume confirmation.
# Goes long when price retraces to L3 support in uptrend (price > 1w EMA50) with volume surge.
# Goes short when price retraces to H3 resistance in downtrend (price < 1w EMA50) with volume surge.
# Designed for low trade frequency (12-37/year) to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly trend filter: EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla pivot levels from previous day
    # Camarilla: range = (high - low), then levels based on close
    # L3 = close - (high - low) * 1.1/4
    # H3 = close + (high - low) * 1.1/4
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous, use same day
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_range = prev_high - prev_low
    l3 = prev_close - camarilla_range * 1.1 / 4
    h3 = prev_close + camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    
    # Volume confirmation (20-period average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price moves below L3 or trend changes
            if close[i] < l3_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above H3 or trend changes
            if close[i] > h3_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: price near L3 support in uptrend (within 0.5% of L3)
                if weekly_uptrend and abs(close[i] - l3_aligned[i]) / l3_aligned[i] < 0.005:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price near H3 resistance in downtrend (within 0.5% of H3)
                elif weekly_downtrend and abs(close[i] - h3_aligned[i]) / h3_aligned[i] < 0.005:
                    position = -1
                    signals[i] = -0.25
    
    return signals
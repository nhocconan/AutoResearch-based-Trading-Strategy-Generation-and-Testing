#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_spike_v1
# Hypothesis: Daily Camarilla pivot levels (L3/H3) with volume spike and weekly trend filter.
# Works in bull/bear: price reverts to mean in range (Camarilla) but only trades with weekly trend.
# Weekly EMA filter avoids counter-trend trades. Volume spike confirms institutional interest.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "1d_camarilla_pivot_volume_spike_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA trend filter (21)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily Camarilla levels (based on previous day)
    # Calculate for each day using previous day's OHLC
    camarilla_H3 = np.full(n, np.nan)
    camarilla_L3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's data
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Camarilla calculation
        range_val = prev_high - prev_low
        camarilla_H3[i] = prev_close + range_val * 1.1 / 4
        camarilla_L3[i] = prev_close - range_val * 1.1 / 4
    
    # Volume filter: volume > 2.0x 20-day average
    vol_ma = np.full(n, np.nan)
    vol_ma[20:] = pd.Series(volume).rolling(window=20, min_periods=20).mean()[20:].values
    
    # Start from sufficient lookback
    start_idx = 21
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_H3[i]) or np.isnan(camarilla_L3[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price reaches L3 (mean reversion target) or trend fails
            if close[i] <= camarilla_L3[i] or close[i] < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price reaches H3 (mean reversion target) or trend fails
            if close[i] >= camarilla_H3[i] or close[i] > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price below L3 with weekly uptrend and volume spike
            if close[i] < camarilla_L3[i] and close[i] > ema_21_1w_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price above H3 with weekly downtrend and volume spike
            elif close[i] > camarilla_H3[i] and close[i] < ema_21_1w_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals
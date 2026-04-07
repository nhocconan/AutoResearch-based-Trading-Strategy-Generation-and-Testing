#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Uses weekly EMA40 for trend direction and daily Camarilla pivot levels (L3/H3) for mean-reversion entries. Enters long when price touches L3 in weekly uptrend, short when price touches H3 in weekly downtrend. Requires volume > 1.5x 20-period average for confirmation. Uses discrete position sizing (0.25) to minimize churn. Designed to work in both bull (trend continuation after pullback to pivot) and bear (counter-trend bounces within larger trend) markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly EMA40 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_40_1w = pd.Series(close_1w).ewm(span=40, min_periods=40, adjust=False).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Daily pivot points for Camarilla levels (based on previous day)
    pivot = (high + low + close) / 3.0
    range_hl = high - low
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to have previous day's pivot
        # Skip if data not available
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price crosses above pivot (mean reversion complete) or trend changes
            if close[i] > pivot[i] or close[i] < ema_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below pivot (mean reversion complete) or trend changes
            if close[i] < pivot[i] or close[i] > ema_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches L3 in weekly uptrend
                if (close[i] <= L3[i] and 
                    close[i] > ema_40_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches H3 in weekly downtrend
                elif (close[i] >= H3[i] and 
                      close[i] < ema_40_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals
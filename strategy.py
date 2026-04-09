#!/usr/bin/env python3
# 1d_weekly_camarilla_pivot_volume_v1
# Hypothesis: On 1d timeframe, price reversals at weekly Camarilla pivot levels (H3/L3) with volume confirmation
# provide high-probability entries in both bull and bear markets. Weekly HTF trend filter (1w EMA21) prevents
# counter-trend trading. Exits occur when price returns to the weekly pivot (mean reversion) or trend reverses.
# Target: 15-25 trades/year (60-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for trend filter and Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Weekly Camarilla pivot levels (based on prior week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    h3 = pivot + range_1w * 1.1 / 2
    l3 = pivot - range_1w * 1.1 / 2
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_1w_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches L3 (mean reversion) OR weekly trend turns bearish
            if close[i] <= l3_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches H3 (mean reversion) OR weekly trend turns bullish
            if close[i] >= h3_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long setup: price touches L3 with bullish weekly trend
                if close[i] <= l3_aligned[i] and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short setup: price touches H3 with bearish weekly trend
                elif close[i] >= h3_aligned[i] and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals
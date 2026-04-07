#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_v1
Hypothesis: On 12h timeframe, use 1-day Camarilla pivot levels for trend context and 1-week EMA for higher timeframe trend filter.
Long when price > H3 and above 1w EMA with volume confirmation; short when price < L3 and below 1w EMA with volume.
Exit when price crosses the opposite H/L level or trend changes. Targets 15-30 trades/year (60-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_v1"
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
    
    # 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    H2 = prev_close + 0.5 * (prev_high - prev_low)
    H1 = prev_close + 0.25 * (prev_high - prev_low)
    L1 = prev_close - 0.25 * (prev_high - prev_low)
    L2 = prev_close - 0.5 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align all levels to 12h timeframe (shifted by 1 day for lookback)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    H2_12h = align_htf_to_ltf(prices, df_1d, H2)
    H1_12h = align_htf_to_ltf(prices, df_1d, H1)
    L1_12h = align_htf_to_ltf(prices, df_1d, L1)
    L2_12h = align_htf_to_ltf(prices, df_1d, L2)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1-week EMA for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False).mean().values
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or np.isnan(ema_1w_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H1 (weaken bullish structure) or trend turns bearish
            if close[i] < H1_12h[i] or close[i] < ema_1w_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above L1 (weaken bearish structure) or trend turns bullish
            if close[i] > L1_12h[i] or close[i] > ema_1w_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must be present for any entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price above H3 and above 1w EMA (bullish breakout with trend)
            if close[i] > H3_12h[i] and close[i] > ema_1w_12h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price below L3 and below 1w EMA (bearish breakdown with trend)
            elif close[i] < L3_12h[i] and close[i] < ema_1w_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
"""
12h_1d_Camarilla_Breakout_Pullback_v1
Hypothesis: In any market regime (bull/bear/range), price tends to pull back to Camarilla pivot levels (especially H3/L3) before continuing the trend. We enter on pullbacks to H3/L3 with volume confirmation in the direction of the 1d trend (EMA50). Stops are placed beyond H4/L4. Designed for low trade frequency and high win rate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Breakout_Pullback_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from previous day
    # Pivot = (H + L + C)/3
    # Range = H - L
    # H4 = P + 1.1*R/2, L4 = P - 1.1*R/2
    # H3 = P + 1.1*R/4, L3 = P - 1.1*R/4
    # H2 = P + 1.1*R/6, L2 = P - 1.1*R/6
    # H1 = P + 1.1*R/12, L1 = P - 1.1*R/12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    h4_1d = pivot_1d + 1.1 * range_1d / 2.0
    l4_1d = pivot_1d - 1.1 * range_1d / 2.0
    h3_1d = pivot_1d + 1.1 * range_1d / 4.0
    l3_1d = pivot_1d - 1.1 * range_1d / 4.0
    
    # Align to 12h timeframe (use previous day's levels)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12H VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Need enough lookback
        # Skip if not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Pullback to H3/L3 with volume confirmation
        near_h3 = abs(close[i] - h3_1d_aligned[i]) / close[i] < 0.005  # Within 0.5%
        near_l3 = abs(close[i] - l3_1d_aligned[i]) / close[i] < 0.005
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: only long if price > EMA50, short if price < EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        long_setup = near_l3 and volume_confirm and uptrend
        short_setup = near_h3 and volume_confirm and downtrend
        
        # Exit when price reaches H4/L4 (stop) or returns to pivot (target)
        long_exit = close[i] >= h4_1d_aligned[i] or close[i] <= pivot_1d[i]  # Stop or target
        short_exit = close[i] <= l4_1d_aligned[i] or close[i] >= pivot_1d[i]  # Stop or target
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif long_exit and position == 1:
            position = 0
            signals[i] = 0.0
        elif short_exit and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
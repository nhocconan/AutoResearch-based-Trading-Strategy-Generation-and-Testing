#!/usr/bin/env python3
"""
6h_Weekly_Pivot_PriceAction
Hypothesis: Weekly pivot points (calculated from prior week) act as strong support/resistance.
Price rejection at weekly R2/S2 with confirmation from daily trend (EMA34) captures high-probability
reversals in both bull and bear markets. Weekly pivot levels provide structure that works across
regimes, reducing false signals. Target: 15-25 trades/year on 6H timeframe.
"""

name = "6h_Weekly_Pivot_PriceAction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (prior week's data)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points from prior week
    # P = (H+L+C)/3, R2 = P + (H-L), S2 = P - (H-L)
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    weekly_p = (h_1w + l_1w + c_1w) / 3.0
    weekly_r2 = weekly_p + (h_1w - l_1w)
    weekly_s2 = weekly_p - (h_1w - l_1w)
    
    # Align weekly pivots to 6h chart (wait for weekly bar to close)
    weekly_p_aligned = align_htf_to_ltf(prices, df_1w, weekly_p)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Get daily trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if position == 0:
            # LONG: Rejection at weekly S2 with daily uptrend and volume
            if (low[i] <= weekly_s2_aligned[i] and 
                close[i] > weekly_s2_aligned[i] and  # bounce off S2
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Rejection at weekly R2 with daily downtrend and volume
            elif (high[i] >= weekly_r2_aligned[i] and 
                  close[i] < weekly_r2_aligned[i] and  # rejection at R2
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot or trend weakens
            if (close[i] >= weekly_p_aligned[i]) or \
               (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot or trend weakens
            if (close[i] <= weekly_p_aligned[i]) or \
               (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
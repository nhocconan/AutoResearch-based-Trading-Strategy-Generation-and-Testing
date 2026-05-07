#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Combo
# Hypothesis: Combines Camarilla pivot points (R1/S1) from daily data with 1d EMA trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal/breakout zones. Breakout of R1/S1 with volume and trend alignment
# captures institutional flow. Works in both bull/bear markets by following the 1d trend direction.
# Target: 20-35 trades/year per symbol to minimize fee drag while maintaining edge.

timeframe = "4h"
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Combo"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day (using typical price)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Where C = (H+L+C)/3 of previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price of previous day
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    # Range of previous day
    range_1d = high_1d - low_1d
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = typical_price_1d + range_1d * 1.1 / 12
    camarilla_s1 = typical_price_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter (proven effective in backtests)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2x average volume (12-period = 2 days on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 12)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume, and 1d trend is bullish (price > EMA34)
            if (high[i] > camarilla_r1_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 with volume, and 1d trend is bearish (price < EMA34)
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S1 (mean reversion to pivot)
            if low[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R1 (mean reversion to pivot)
            if high[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
# 6h_Weekly_Pivot_Reversal_1dTrend_Volume
# Hypothesis: Reversals at weekly pivot support/resistance levels with daily trend filter and volume confirmation
# - Weekly pivot levels provide key weekly support/resistance from prior week's price action
# - Reversal signals at S1/R1 levels when price rejects these levels with strong volume
# - Daily trend filter (EMA34) ensures reversals align with higher timeframe momentum
# - Volume confirmation (1.5x average) reduces false reversals
# - Works in both bull (reversals at support in uptrend) and bear (reversals at resistance in downtrend)
# - Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range for 6h timeframe

#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Weekly pivot levels from previous week (standard formula)
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    w_pivot = (w_high + w_low + w_close) / 3
    w_range = w_high - w_low
    w_r1 = w_pivot + (w_range * 1.1 / 12)
    w_s1 = w_pivot - (w_range * 1.1 / 12)
    
    # Align weekly pivot levels to 6h timeframe
    w_r1_6h = align_htf_to_ltf(prices, df_1w, w_r1)
    w_s1_6h = align_htf_to_ltf(prices, df_1w, w_s1)
    w_pivot_6h = align_htf_to_ltf(prices, df_1w, w_pivot)
    
    # Daily EMA34 for trend filter
    d_close = df_1d['close'].values
    ema_34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(w_r1_6h[i]) or np.isnan(w_s1_6h[i]) or 
            np.isnan(ema_34_6h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(w_pivot_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: reversal at S1 support in daily uptrend with volume
            if close[i] < w_s1_6h[i] and low[i] <= w_s1_6h[i] * 1.001 and close[i] > w_s1_6h[i] and close[i] > ema_34_6h[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: reversal at R1 resistance in daily downtrend with volume
            elif close[i] > w_r1_6h[i] and high[i] >= w_r1_6h[i] * 0.999 and close[i] < w_r1_6h[i] and close[i] < ema_34_6h[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches pivot or trend reverses
            if close[i] >= w_pivot_6h[i] or close[i] < ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches pivot or trend reverses
            if close[i] <= w_pivot_6h[i] or close[i] > ema_34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
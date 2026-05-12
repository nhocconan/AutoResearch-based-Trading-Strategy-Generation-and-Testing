# 6h_PivotRange_Reversal_1dTrendFilter
# Strategy: Mean reversion within 1d CPR (Central Pivot Range) with 1d trend filter
# Works in bull/bear by fading extremes within daily range when trend is clear
# Target: 50-150 trades over 4 years via tight CPR boundaries and trend alignment
# Uses 1d CPR (Central Pivot Range) and 1d EMA50 for trend filter
# Entry: Price at CPR boundary + contrarian signal + 1d trend alignment
# Exit: Opposite CPR boundary or trend reversal

name = "6h_PivotRange_Reversal_1dTrendFilter"
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
    
    # Load 1d data once for CPR and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily OHLC for CPR calculation (TC, BC, PP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate CPR: TC = (H+L)/2, BC = (H+L+C)/3, PP = (H+L+2C)/4
    tc = (high_1d + low_1d) / 2  # Top Central
    bc = (high_1d + low_1d + close_1d_vals) / 3  # Bottom Central
    pp = (high_1d + low_1d + 2 * close_1d_vals) / 4  # Pivot Point
    
    # CPR boundaries: TC (upper) and BC (lower)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc)
    
    # Signal definitions
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tc_aligned[i]) or np.isnan(bc_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price at BC (CPR lower) + below EMA50 (contrarian in downtrend) 
            # OR price at TC (CPR upper) + above EMA50 (contrarian in uptrend)
            # Actually, we want to fade extremes: buy near BC in uptrend, sell near TC in downtrend
            if close[i] <= bc_aligned[i] * 1.001 and close[i] >= bc_aligned[i] * 0.999:  # at BC
                if close[i] > ema_50_1d_aligned[i]:  # in uptrend, buy dips
                    signals[i] = 0.25
                    position = 1
            elif close[i] >= tc_aligned[i] * 0.999 and close[i] <= tc_aligned[i] * 1.001:  # at TC
                if close[i] < ema_50_1d_aligned[i]:  # in downtrend, sell rallies
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches TC (CPR upper) or trend flips
            if close[i] >= tc_aligned[i] * 0.999 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches BC (CPR lower) or trend flips
            if close[i] <= bc_aligned[i] * 1.001 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
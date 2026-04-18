# 12h_Top_Tier_Pivot_Breakout_Volume_Trend
# Hypothesis: Pivot point breakouts (R1/S1) with volume confirmation and trend alignment (weekly EMA34) capture institutional flow with minimal whipsaw.
# Works in bull/bear: Pivots adapt to volatility, volume confirms institutional participation, weekly trend filter avoids countertrend traps.
# Target: 15-25 trades/year to minimize fee drag while capturing major moves.

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
    
    # Daily Pivot Points (calculate once per day, align to 12h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 12h timeframe (with proper delay for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = vol_ratio[i] > 1.5
        
        # Long conditions: price breaks above R1 with volume and weekly uptrend
        if price > r1_aligned[i] and vol_conf and price > ema34_1w_aligned[i]:
            if position <= 0:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        
        # Short conditions: price breaks below S1 with volume and weekly downtrend
        elif price < s1_aligned[i] and vol_conf and price < ema34_1w_aligned[i]:
            if position >= 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        
        # Exit conditions: price returns to pivot or volume dries up
        elif position == 1 and (price < pivot_aligned[i] or vol_ratio[i] < 1.0):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (price > pivot_aligned[i] or vol_ratio[i] < 1.0):
            signals[i] = 0.0
            position = 0
        
        # Hold position
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_Top_Tier_Pivot_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0
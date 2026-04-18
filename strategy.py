#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Breakout_With_Trend_Filter
Hypothesis: Weekly (1w) CPR (Central Pivot Range) levels act as strong support/resistance.
Breakout above weekly CPR high with volume > 1.5x 20-day average and price above weekly EMA20 = long.
Breakdown below weekly CPR low with volume confirmation and price below weekly EMA20 = short.
Designed for daily timeframe with ~10-25 trades/year to minimize fee drift and work in both bull and bear via trend filter.
"""

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
    
    # Weekly data for CPR and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly CPR (Central Pivot Range)
    # TC = (H + L + 2C) / 4  (Top Central)
    # BC = (H + L) / 2        (Bottom Central)
    # PV = (H + L + C) / 3    (Pivot Point)
    tc_1w = (high_1w + low_1w + 2 * close_1w) / 4
    bc_1w = (high_1w + low_1w) / 2
    pv_1w = (high_1w + low_1w + close_1w) / 3
    
    # CPR boundaries: higher of TC/PV as top, lower of BC/PV as bottom
    cpr_top = np.maximum(tc_1w, pv_1w)
    cpr_bottom = np.minimum(bc_1w, pv_1w)
    
    # Align weekly CPR to daily timeframe (wait for weekly bar close)
    cpr_top_aligned = align_htf_to_ltf(prices, df_1w, cpr_top)
    cpr_bottom_aligned = align_htf_to_ltf(prices, df_1w, cpr_bottom)
    
    # Weekly EMA trend filter (20-period)
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily volume filter: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(cpr_top_aligned[i]) or np.isnan(cpr_bottom_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        cpr_top_val = cpr_top_aligned[i]
        cpr_bottom_val = cpr_bottom_aligned[i]
        ema_trend = ema_20_1w_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above weekly CPR top with volume in uptrend
            if price > cpr_top_val and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly CPR bottom with volume in downtrend
            elif price < cpr_bottom_val and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below weekly CPR bottom or trend reverses
            if price < cpr_bottom_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above weekly CPR top or trend reverses
            if price > cpr_top_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Pivot_Breakout_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0
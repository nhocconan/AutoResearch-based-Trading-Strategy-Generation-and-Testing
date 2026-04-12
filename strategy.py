#!/usr/bin/env python3
"""
12h_1d_Volume_Weighted_Pullback
Hypothesis: Combines 12h price pullback to dynamic support/resistance (20 EMA) with 1d volume-weighted momentum confirmation.
Enters long when price pulls back to EMA20 during uptrend with rising volume, short when price rallies to EMA20 during downtrend with falling volume.
Designed for low trade frequency by requiring alignment of price action, trend, and volume confirmation.
Works in bull via buying dips in uptrend, in bear by selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Volume_Weighted_Pullback"
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
    
    # === DAILY DATA (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA20 for trend
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Daily volume EMA20 for momentum
    volume_1d_series = pd.Series(volume_1d)
    vol_ema20_1d = volume_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20_1d)
    
    # Daily volume ratio (current vs EMA20)
    vol_ratio_1d = volume_1d / np.where(vol_ema20_1d == 0, 1, vol_ema20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # 12h price relative to daily EMA20
        price_near_ema = abs(close[i] - ema20_1d_aligned[i]) / ema20_1d_aligned[i] < 0.015  # within 1.5%
        
        # Daily volume momentum
        vol_expanding = vol_ratio_1d_aligned[i] > 1.1  # volume above average
        vol_contracting = vol_ratio_1d_aligned[i] < 0.9  # volume below average
        
        # Price trend (using 12h close vs prior)
        price_up = close[i] > close[i-1]
        price_down = close[i] < close[i-1]
        
        # Entry conditions
        long_setup = price_near_ema and price_up and vol_expanding
        short_setup = price_near_ema and price_down and vol_contracting
        
        # Exit when conditions reverse or stop conditions
        exit_long = not (price_near_ema and price_up and vol_expanding)
        exit_short = not (price_near_ema and price_down and vol_contracting)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals
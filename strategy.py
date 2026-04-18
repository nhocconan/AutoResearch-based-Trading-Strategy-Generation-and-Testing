#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Confirmation and Daily Trend Filter
Hypothesis: In both bull and bear markets, price breaking above R1 or below S1 of daily Camarilla pivots
with volume > 1.5x average and aligned with daily EMA34 trend captures strong momentum.
Daily EMA34 filter prevents counter-trend trades, reducing whipsaws in sideways markets.
Target: 20-30 trades/year to minimize fee drag while capturing major moves.
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
    
    # Daily EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily high/low/close for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels: R1, S1
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    rng = high_1d - low_1d
    r1 = close_1d + 1.1 * rng / 12
    s1 = close_1d - 1.1 * rng / 12
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 with volume and price > daily EMA34 (uptrend)
            if price > r1_val and vol_conf and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and price < daily EMA34 (downtrend)
            elif price < s1_val and vol_conf and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to daily EMA34 or volume dries up
            if price < ema34 or vol_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to daily EMA34 or volume dries up
            if price > ema34 or vol_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_DailyEMA34"
timeframe = "12h"
leverage = 1.0
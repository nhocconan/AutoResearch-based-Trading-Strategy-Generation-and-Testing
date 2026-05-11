#!/usr/bin/env python3
"""
1D_1W_Camarilla_Pivot_Breakout_Volume
Hypothesis: Price breaking above/below weekly Camarilla pivot-derived resistance/support (R4/S4) with volume confirmation, filtered by weekly trend (price > EMA50 weekly). Weekly pivots capture institutional levels; volume confirms participation. Weekly trend filter avoids counter-trend whipsaws. Designed for low frequency (10-25 trades/year) to work in both bull (breakouts) and bear (mean reversion at extremes) markets.
"""

name = "1D_1W_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Weekly Camarilla Pivot Points (R4, S4) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pp_w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly range
    range_1w = high_1w - low_1w
    # Camarilla levels: R4 = close + range * 1.1/2, S4 = close - range * 1.1/2
    r4_w = close_1w + (range_1w * 1.1 / 2)
    s4_w = close_1w - (range_1w * 1.1 / 2)
    
    # Align weekly levels to daily timeframe (using previous week's levels)
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # --- Weekly EMA50 for trend filter ---
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Volume Spike (daily) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_w_aligned[i]) or 
            np.isnan(s4_w_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma.values[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: 
        # Long: Price > weekly R4 AND volume spike AND above weekly EMA50
        # Short: Price < weekly S4 AND volume spike AND below weekly EMA50
        long_entry = (close[i] > r4_w_aligned[i]) and \
                     vol_spike[i] and \
                     (close[i] > ema_50_1w_aligned[i])
        
        short_entry = (close[i] < s4_w_aligned[i]) and \
                      vol_spike[i] and \
                      (close[i] < ema_50_1w_aligned[i])
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: 
            # Long: Price crosses below weekly pivot OR below weekly EMA50
            # Short: Price crosses above weekly pivot OR above weekly EMA50
            if position == 1:
                pp_w_aligned = align_htf_to_ltf(prices, df_1w, pp_w)
                if (close[i] < pp_w_aligned[i]) or \
                   (close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                pp_w_aligned = align_htf_to_ltf(prices, df_1w, pp_w)
                if (close[i] > pp_w_aligned[i]) or \
                   (close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
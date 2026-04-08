#!/usr/bin/env python3
# 6h_1d_camarilla_pivot_breakout_v1
# Hypothesis: Trade Camarilla pivot breakouts on 6h timeframe with daily trend filter.
# Uses daily Camarilla levels (H3/L3 for breakout, H4/L4 for reversal) and 6h price action.
# In bull markets: buy breakouts above H3 with daily uptrend. In bear markets: sell breakdowns below L3 with daily downtrend.
# Includes volume confirmation to filter false breakouts. Target: 20-40 trades/year on 6h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # H2 = close + 0.75*(high-low), L2 = close - 0.75*(high-low)
    # H1 = close + 0.5*(high-low), L1 = close - 0.5*(high-low)
    range_1d = high_1d - low_1d
    H4 = close_1d + 1.5 * range_1d
    L4 = close_1d - 1.5 * range_1d
    H3 = close_1d + 1.125 * range_1d
    L3 = close_1d - 1.125 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Daily trend filter: EMA20 vs EMA50
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    daily_uptrend = ema20_aligned > ema50_aligned
    daily_downtrend = ema20_aligned < ema50_aligned
    
    # Volume confirmation: 6h volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(ema20_aligned[i]) or np.isnan(ema50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: breakdown below L3 OR reversal at H4
            if close[i] < L3_aligned[i] or close[i] > H4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: breakout above H3 OR reversal at L4
            if close[i] > H3_aligned[i] or close[i] < L4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above H3 with daily uptrend and volume surge
            if close[i] > H3_aligned[i] and daily_uptrend[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below L3 with daily downtrend and volume surge
            elif close[i] < L3_aligned[i] and daily_downtrend[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals
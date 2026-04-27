#!/usr/bin/env python3
"""
6h_WickReversal_Volume_Trend
Hypothesis: Price rejection at key levels (long upper/lower wicks) with volume confirmation and trend filter captures reversals. Works in bull/bear markets by fading overextended moves. Uses 1w trend filter for major direction and 1d for swing structure. Targets 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for major trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data for swing structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly trend: EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily swing high/low for Wick detection (using 5-period lookback)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    roll_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    roll_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    
    # Align daily swing levels to 6h
    roll_high_aligned = align_htf_to_ltf(prices, df_1d, roll_high)
    roll_low_aligned = align_htf_to_ltf(prices, df_1d, roll_low)
    
    # Volume confirmation: volume > 1.5 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(roll_high_aligned[i]) or 
            np.isnan(roll_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Wick conditions: long upper wick (selling pressure) or long lower wick (buying pressure)
        body_size = abs(close[i] - open_price) if 'open_price' in locals() else abs(close[i] - close[i-1]) if i > 0 else 0
        upper_wick = high[i] - max(close[i], open_price if i > 0 else close[i])
        lower_wick = min(close[i], open_price if i > 0 else close[i]) - low[i]
        # Simplified: use high-low range and close position
        true_range = high[i] - low[i]
        if true_range == 0:
            signals[i] = 0.0
            continue
        close_position = (close[i] - low[i]) / true_range  # 0 = low close, 1 = high close
        
        ema_trend = ema50_1w_aligned[i]
        swing_high = roll_high_aligned[i]
        swing_low = roll_low_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: rejection at swing low (long lower wick) with volume and uptrend
            if close_position < 0.3 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: rejection at swing high (long upper wick) with volume and downtrend
            elif close_position > 0.7 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: rejection at swing high or trend turns down
            if close_position > 0.7 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: rejection at swing low or trend turns up
            if close_position < 0.3 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WickReversal_Volume_Trend"
timeframe = "6h"
leverage = 1.0
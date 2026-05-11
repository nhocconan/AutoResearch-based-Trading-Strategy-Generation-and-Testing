#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Price breaking above/below daily Camarilla R1/S1 levels with volume confirmation and daily trend filter (price > EMA50). 
Camarilla levels from daily timeframe provide institutional support/resistance. Volume confirms participation. 
Trend filter avoids counter-trwhipsaws. Designed for 4h timeframe with low frequency (20-50 trades/year) 
to work in bull (breakouts) and bear (mean reversion at extremes) markets.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Levels (R1, S1) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day: use same day's values (will be NaN until second day due to min_periods)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla R1 and S1 using previous day's data
    r1_1d = prev_close + 1.1 * (prev_high - prev_low) / 12
    s1_1d = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align daily levels to 4h timeframe (using previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- Daily EMA50 for trend filter ---
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volume Spike (4h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
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
        # Long: Price > daily R1 AND volume spike AND above daily EMA50
        # Short: Price < daily S1 AND volume spike AND below daily EMA50
        long_entry = (close[i] > r1_1d_aligned[i]) and \
                     vol_spike[i] and \
                     (close[i] > ema_50_1d_aligned[i])
        
        short_entry = (close[i] < s1_1d_aligned[i]) and \
                      vol_spike[i] and \
                      (close[i] < ema_50_1d_aligned[i])
        
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
            # Long: Price crosses below daily S1 OR below daily EMA50
            # Short: Price crosses above daily R1 OR above daily EMA50
            if position == 1:
                if (close[i] < s1_1d_aligned[i]) or \
                   (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (close[i] > r1_1d_aligned[i]) or \
                   (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals
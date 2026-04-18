#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Tight_V1
Hypothesis: Use daily pivot R1/S1 for directional bias, 12h for entry with volume confirmation.
Long when price breaks above daily R1 with volume > 1.5x average during active session (08-20 UTC).
Short when price breaks below daily S1 with volume > 1.5x average during active session.
Tighten: require price to close outside pivot band for 2 consecutive 12h bars to avoid false breaks.
Position size 0.25. Target ~25 trades/year per symbol (100 total over 4 years) to minimize fee drag.
Works in bull/bear via session timing and volume confirmation.
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's OHLC for standard pivot calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Volatility filter: avoid extreme volatility (stop hunting)
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(np.roll(close_1d, 1) - low_1d)
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma_50 = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values
    
    # Align all daily data to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    consecutive_outside = 0  # count of consecutive bars outside pivot band
    
    start_idx = 50  # need enough for ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_20_aligned[i]) or np.isnan(atr_ma_50_aligned[i])):
            signals[i] = 0.0
            consecutive_outside = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Volatility filter: avoid extreme volatility
        vol_filter = atr_20_aligned[i] < atr_ma_50_aligned[i] * 2 if not np.isnan(atr_ma_50_aligned[i]) else False
        
        # Only trade during active session
        in_session = session_mask[i]
        
        # Check if price is outside pivot band
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        outside_band = price_above_r1 or price_below_s1
        
        # Count consecutive bars outside band
        if outside_band:
            consecutive_outside += 1
        else:
            consecutive_outside = 0
        
        # Require 2 consecutive bars outside band to confirm breakout
        confirmed_breakout = consecutive_outside >= 2
        
        if position == 0:
            # Long: price breaks above R1 with volume and volatility filter during session
            if price_above_r1 and vol_confirm and vol_filter and in_session and confirmed_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and volatility filter during session
            elif price_below_s1 and vol_confirm and vol_filter and in_session and confirmed_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below pivot or volatility spike or outside session
            if close[i] < pivot_aligned[i] or not vol_filter or not in_session:
                signals[i] = -0.25  # reverse to short
                position = -1
                consecutive_outside = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot or volatility spike or outside session
            if close[i] > pivot_aligned[i] or not vol_filter or not in_session:
                signals[i] = 0.25  # reverse to long
                position = 1
                consecutive_outside = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Tight_V1"
timeframe = "12h"
leverage = 1.0
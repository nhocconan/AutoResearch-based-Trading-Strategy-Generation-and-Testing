#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R(14) mean reversion + 1d EMA50 trend filter + volume spike confirmation
- Williams %R identifies overbought/oversold conditions for mean reversion entries
- 1d EMA50 establishes primary trend direction (long only in uptrend, short only in downtrend)
- Volume spike (2.5x 20-period MA) confirms institutional participation at reversal points
- Fixed 4-bar holding period reduces whipsaw and controls trade frequency
- Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend)
- Aligns with proven patterns: Williams %R + volume + trend filter shows edge in mean reversion
- Target: 25-35 trades/year per symbol (~100-140 total over 4 years)
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Get 4h data for primary timeframe (Williams %R, volume)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 4h
    def williams_r(high_arr, low_arr, close_arr, window=14):
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_arr) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_14_4h = williams_r(high_4h, low_4h, close_4h, 14)
    
    # Volume average (20-period) on 4h
    volume_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    wr_14_aligned = align_htf_to_ltf(prices, df_4h, wr_14_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    bars_since_entry = 0  # For fixed holding period
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(wr_14_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        wr_val = wr_14_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        # Increment bars since entry
        if position != 0:
            bars_since_entry += 1
        
        if position == 0:
            # Look for mean reversion entries with volume confirmation and trend alignment
            # Long: Williams %R oversold (< -80) + volume spike + price > 1d EMA50 (uptrend)
            if wr_val < -80.0 and vol > 2.5 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Williams %R overbought (> -20) + volume spike + price < 1d EMA50 (downtrend)
            elif wr_val > -20.0 and vol > 2.5 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position != 0:
            # Fixed 4-bar holding period exit
            if bars_since_entry >= 4:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR14_1dEMA50_VolumeSpike_FixedHold"
timeframe = "4h"
leverage = 1.0
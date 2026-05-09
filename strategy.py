# 12H_Camarilla_R1_S1_Breakout_1dTrend_Volume - Optimized for lower trade frequency
# Strategy: Daily Camarilla R1/S1 breakout with 1d trend filter and volume confirmation
# Timeframe: 12h to reduce trade frequency vs 4h
# Entry: Trend (price > 1d EMA50) + price breaks above/below R1/S1 + volume surge (1.5x 12h volume EMA20)
# Exit: Trend reversal or price breaks opposite level
# Position sizing: 0.25 (25%) to manage drawdown
# Expected trades: 20-40/year to stay within limits and avoid fee drag

#!/usr/bin/env python3
name = "12H_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = C + 1.1*(H-L)/12, S1 = C - 1.1*(H-L)/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        camarilla_r1[i] = prev_close + 1.1 * range_ / 12
        camarilla_s1[i] = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 12h data for volume confirmation
    volume_12h = volume  # Already at 12h timeframe
    vol_ema20_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ema20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        # Downtrend: price below 1d EMA50
        downtrend = close[i] < ema50_1d_aligned[i]
        # Volume surge: current volume > 1.5x 12h volume EMA20 (moderate threshold)
        volume_surge = volume[i] > vol_ema20_12h[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price breaks above Camarilla R1 + volume surge
            if uptrend and close[i] > camarilla_r1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price breaks below Camarilla S1 + volume surge
            elif downtrend and close[i] < camarilla_s1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price breaks below Camarilla S1
            if not uptrend or close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price breaks above Camarilla R1
            if not downtrend or close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
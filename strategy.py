#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with volume confirmation and 1-week trend filter.
# Long when price breaks above daily pivot-based R1 with volume > 1.5x 24-period average and weekly close > weekly open.
# Short when price breaks below daily pivot-based S1 with volume > 1.5x 24-period average and weekly close < weekly open.
# Exit when price returns to daily pivot (PP).
# Uses weekly trend filter to avoid counter-trend trades, volume for conviction, and Camarilla levels for structure.
# Designed for ~15-30 trades/year per symbol.
name = "12h_Camarilla_R1S1_Breakout_Volume_WeeklyTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily Camarilla levels: PP, R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align daily Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Weekly trend: bullish if weekly close > weekly open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w  # True for bullish week
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5  # Convert back to boolean
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume surge and weekly bullish
            if close_val > r1_val and vol_filter and weekly_bull:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume surge and weekly bearish
            elif close_val < s1_val and vol_filter and not weekly_bull:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to or below pivot point
            if close_val <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to or above pivot point
            if close_val >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
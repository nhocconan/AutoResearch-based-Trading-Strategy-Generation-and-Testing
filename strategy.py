#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above 1d R1 AND 1d volume > 1.5x 20-period average AND 1w close > 1w open (bullish week)
# Short when price breaks below 1d S1 AND 1d volume > 1.5x 20-period average AND 1w close < 1w open (bearish week)
# Exit when price returns to 1d pivot point (PP)
# Uses Camarilla for precise intraday levels, volume for conviction, weekly trend to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).
name = "12h_Camarilla_R1S1_Volume_WeeklyTrend"
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
    
    # Get 1d data for Camarilla pivot and volume
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pp = (high_1d + low_1d + close_1d) / 3
    r1 = pp + (high_1d - low_1d) * 1.1 / 12
    s1 = pp - (high_1d - low_1d) * 1.1 / 12
    
    # Get 1d average volume for confirmation
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for trend filter (bullish/bearish week)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w  # True for bullish week
    
    # Align all 1d and 1w arrays to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure 1d indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_1d_aligned[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5  # Convert back to boolean
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above R1 + volume + bullish weekly trend
            if price > r1_val and vol_confirm and weekly_bull:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + volume + bearish weekly trend
            elif price < s1_val and vol_confirm and not weekly_bull:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point
            if price <= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point
            if price >= pp_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
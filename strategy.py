#!/usr/bin/env python3
"""
1d_WeeklyTrend_Filtered_Breakout_v1
Hypothesis: Buy when price breaks above weekly Donchian high (20-period) with volume confirmation in uptrend;
sell/short when price breaks below weekly Donchian low with volume confirmation in downtrend.
Uses weekly trend filter to avoid counter-trend trades. Designed for low frequency (<25 trades/year) to minimize fee drag.
Works in bull markets via trend-following breaks and in bear markets via short-side breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    weekly_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA50 on weekly close
    close_1w = df_1w['close'].values
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align weekly indicators to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need weekly Donchian (20), weekly EMA50 (50), volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_high_val = weekly_high_aligned[i]
        weekly_low_val = weekly_low_aligned[i]
        weekly_ema50_val = weekly_ema50_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs weekly EMA50
            uptrend = close_val > weekly_ema50_val
            downtrend = close_val < weekly_ema50_val
            
            if uptrend and vol_conf:
                # Long: break above weekly Donchian high with volume
                if close_val > weekly_high_val:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below weekly Donchian low with volume
                if close_val < weekly_low_val:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters weekly channel or trend reversal to downtrend
            if close_val < weekly_high_val:  # Re-enter weekly channel
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters weekly channel or trend reversal to uptrend
            if close_val > weekly_low_val:  # Re-enter weekly channel
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyTrend_Filtered_Breakout_v1"
timeframe = "1d"
leverage = 1.0
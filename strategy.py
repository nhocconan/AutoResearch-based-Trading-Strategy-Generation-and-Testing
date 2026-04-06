#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Filter
Hypothesis: Buy when price breaks above weekly Donchian high with weekly uptrend and volume confirmation; sell when breaks below weekly Donchian low with weekly downtrend and volume confirmation.
Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend and Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Daily data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require above-average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 50  # For EMA50 and Donchian(20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or loss of trend
        if position == 1:  # long position
            # Exit: price breaks below weekly Donchian low OR trend turns down
            if (close[i] <= donch_low_aligned[i] or 
                ema_50_1w_aligned[i] < donch_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above weekly Donchian high OR trend turns up
            if (close[i] >= donch_high_aligned[i] or 
                ema_50_1w_aligned[i] > donch_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with trend and volume
            long_breakout = (close[i] > donch_high_aligned[i])  # Break above Donchian high
            short_breakout = (close[i] < donch_low_aligned[i])  # Break below Donchian low
            
            uptrend = ema_50_1w_aligned[i] > donch_high_aligned[i]  # Trend above Donchian high
            downtrend = ema_50_1w_aligned[i] < donch_low_aligned[i]  # Trend below Donchian low
            
            if long_breakout and uptrend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            elif short_breakout and downtrend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals
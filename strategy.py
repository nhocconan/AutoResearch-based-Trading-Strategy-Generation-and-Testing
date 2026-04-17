#391
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d ADX and 1w Bollinger Bands for trend-following breakouts.
Trade breakouts from Bollinger Bands (20, 2) when weekly ADX > 25 indicates strong trend.
Use daily ADX > 20 to confirm trend alignment between daily and weekly.
In strong uptrends: buy breakouts above upper BB; in strong downtrends: sell breakdowns below lower BB.
Position sizing: 0.25 for entries, 0 for exits.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1w data for Bollinger Bands and ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Bollinger Bands (20, 2) on weekly
    sma_20w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_20w = sma_20w + 2 * std_20w
    lower_bb_20w = sma_20w - 2 * std_20w
    
    # Calculate ADX (14) on weekly
    plus_dm_w = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                         np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm_w = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                          np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm_w = np.concatenate([[0], plus_dm_w])
    minus_dm_w = np.concatenate([[0], minus_dm_w])
    
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3_w = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    plus_di_w = 100 * pd.Series(plus_dm_w).rolling(window=14, min_periods=14).sum().values / atr_w
    minus_di_w = 100 * pd.Series(minus_dm_w).rolling(window=14, min_periods=14).sum().values / atr_w
    dx_w = 100 * np.abs(plus_di_w - minus_di_w) / (plus_di_w + minus_di_w)
    adx_w = pd.Series(dx_w).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on daily
    plus_dm_d = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                         np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm_d = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                          np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm_d = np.concatenate([[0], plus_dm_d])
    minus_dm_d = np.concatenate([[0], minus_dm_d])
    
    tr1_d = high_1d - low_1d
    tr2_d = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3_d = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    plus_di_d = 100 * pd.Series(plus_dm_d).rolling(window=14, min_periods=14).sum().values / atr_d
    minus_di_d = 100 * pd.Series(minus_dm_d).rolling(window=14, min_periods=14).sum().values / atr_d
    dx_d = 100 * np.abs(plus_di_d - minus_di_d) / (plus_di_d + minus_di_d)
    adx_d = pd.Series(dx_d).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 6h
    upper_bb_20w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_20w)
    lower_bb_20w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_20w)
    adx_w_aligned = align_htf_to_ltf(prices, df_1w, adx_w)
    adx_d_aligned = align_htf_to_ltf(prices, df_1d, adx_d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_20w_aligned[i]) or np.isnan(lower_bb_20w_aligned[i]) or 
            np.isnan(adx_w_aligned[i]) or np.isnan(adx_d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend alignment: both daily and weekly ADX > thresholds and same direction
        strong_weekly_trend = adx_w_aligned[i] > 25
        strong_daily_trend = adx_d_aligned[i] > 20
        
        # For simplicity, we use price relative to BB to infer direction
        # In practice, we could check weekly +/- DI but keeping it simple
        if position == 0:
            # Long: breakout above upper BB with strong trend on both timeframes
            if (close[i] > upper_bb_20w_aligned[i] and 
                strong_weekly_trend and strong_daily_trend):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below lower BB with strong trend on both timeframes
            elif (close[i] < lower_bb_20w_aligned[i] and 
                  strong_weekly_trend and strong_daily_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle of BB or trend weakens
            middle_bb = (upper_bb_20w_aligned[i] + lower_bb_20w_aligned[i]) / 2
            if close[i] < middle_bb or adx_w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle of BB or trend weakens
            middle_bb = (upper_bb_20w_aligned[i] + lower_bb_20w_aligned[i]) / 2
            if close[i] > middle_bb or adx_w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wBBands20_2_ADX14_1w_1d"
timeframe = "6h"
leverage = 1.0
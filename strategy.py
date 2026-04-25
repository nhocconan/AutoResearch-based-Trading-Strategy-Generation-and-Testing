#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dRegime_ChopFilter_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 1d trend filter and chop regime filter. 
In trending markets (CHOP < 38.2): buy breakouts above R1 in uptrend, sell breakdowns below S1 in downtrend. 
In ranging markets (CHOP > 61.8): fade extremes - sell near R1, buy near S1. 
Requires volume > 1.3x 20-period average for confirmation. 
Position size: 0.25. 
Target: 75-200 total trades over 4 years = 19-50/year. 
Uses Camarilla structure from daily timeframe for institutional relevance.
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
    
    # Get 1d data for Camarilla levels, trend, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for indicators
        return np.zeros(n)
    
    # Calculate daily EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    hl_range_1d = high_1d - low_1d
    r1_1d = close_1d + (1.1 * hl_range_1d / 12)  # R1 = close + 1.1*(high-low)/12
    s1_1d = close_1d - (1.1 * hl_range_1d / 12)  # S1 = close - 1.1*(high-low)/12
    
    # Align Camarilla levels to match prices index
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate Choppiness Index on 1d (regime filter)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(low_1d[1:] - high_1d[:-1], np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # Align with original arrays
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / 14) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), ATR (14), chop (14)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above daily EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Regime filter based on Choppiness Index
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long setup conditions
            long_breakout = close[i] > r1_aligned[i]
            long_trend_follow = long_breakout and htf_1d_bullish and is_trending and volume_confirm
            long_mean_revert = (close[i] < r1_aligned[i] * 1.005) and htf_1d_bearish and is_ranging and volume_confirm  # Near R1 in range
            
            # Short setup conditions
            short_breakdown = close[i] < s1_aligned[i]
            short_trend_follow = short_breakdown and htf_1d_bearish and is_trending and volume_confirm
            short_mean_revert = (close[i] > s1_aligned[i] * 0.995) and htf_1d_bullish and is_ranging and volume_confirm  # Near S1 in range
            
            if long_trend_follow or long_mean_revert:
                signals[i] = 0.25
                position = 1
            elif short_trend_follow or short_mean_revert:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if (close[i] < s1_aligned[i]) or (not htf_1d_bullish and is_trending) or (chop_aligned[i] > 61.8 and close[i] < r1_aligned[i] * 0.995):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if (close[i] > r1_aligned[i]) or (htf_1d_bullish and is_trending) or (chop_aligned[i] > 61.8 and close[i] > s1_aligned[i] * 1.005):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dRegime_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0
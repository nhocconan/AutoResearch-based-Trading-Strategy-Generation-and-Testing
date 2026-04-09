#!/usr/bin/env python3
# 6h_donchian_weekly_pivot_volume_v1
# Hypothesis: 6h strategy combining Donchian channel breakouts with weekly Camarilla pivot structure and volume confirmation.
# Long when price breaks above 6h Donchian(20) upper band AND is above weekly R3 pivot with volume > 1.8x 20-period average.
# Short when price breaks below 6h Donchian(20) lower band AND is below weekly S3 pivot with volume > 1.8x 20-period average.
# Exit when price returns to the weekly pivot level (mean reversion to equilibrium).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Designed to capture strong breakouts aligned with weekly structure while avoiding false signals in chop.
# Target: 12-30 trades/year (50-120 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for Camarilla pivot levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    pivot_1w = typical_price_1w
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    r2_1w = close_1w + (range_1w * 1.1 / 6)
    s2_1w = close_1w - (range_1w * 1.1 / 6)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # Align all weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(pivot_1w_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume[i] > 1.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to weekly pivot (mean reversion)
            if close[i] <= pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to weekly pivot (mean reversion)
            if close[i] >= pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and weekly pivot filter
            bullish_breakout = (close[i] > highest_high[i]) and volume_confirmed and (close[i] > r3_1w_aligned[i])
            bearish_breakout = (close[i] < lowest_low[i]) and volume_confirmed and (close[i] < s3_1w_aligned[i])
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals
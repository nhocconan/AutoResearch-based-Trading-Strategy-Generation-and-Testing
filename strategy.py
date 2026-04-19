#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day Bollinger Band squeeze breakout and volume confirmation.
# Uses Bollinger Band width percentile to identify low volatility periods (squeeze), then enters
# on breakout above/below the bands with volume confirmation. Works in both bull and bear
# markets by capturing volatility expansion after consolidation. Target: 50-150 total trades
# over 4 years (12-37/year) to minimize fee drag while capturing meaningful moves.
name = "12h_1d_BB_Squeeze_Breakout_Volume"
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
    
    # Get 1d data for Bollinger Bands (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands on 1d timeframe (20-period, 2 std dev)
    bb_period = 20
    bb_std = 2
    sma_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_1d = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_1d + (std_1d * bb_std)
    lower_bb = sma_1d - (std_1d * bb_std)
    bb_width = upper_bb - lower_bb
    
    # Calculate BB width percentile rank (252-period lookback ~1 year)
    lookback = 252
    bb_width_series = pd.Series(bb_width)
    bb_width_rank = bb_width_series.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Squeeze condition: BB width below 20th percentile (low volatility)
    squeeze_condition = bb_width_rank < 20
    
    # Align 1d indicators to 12h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Volume filter: volume > 1.5 * 20-period average on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, lookback, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when squeeze breaks above upper BB with volume
            if squeeze_aligned[i] and close[i] > upper_bb_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short when squeeze breaks below lower BB with volume
            elif squeeze_aligned[i] and close[i] < lower_bb_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to middle (SMA) or breaks below lower BB
            if close[i] < sma_1d[-1] if len(sma_1d) > 0 else False:  # Simplified: exit at SMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to middle (SMA) or breaks above upper BB
            if close[i] > sma_1d[-1] if len(sma_1d) > 0 else False:  # Simplified: exit at SMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
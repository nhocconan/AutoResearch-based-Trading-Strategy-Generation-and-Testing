#!/usr/bin/env python3
"""
6h_1d_1w_Liquidity_Capture
Hypothesis: Uses 1w liquidity pools (weekly high/low) and 1d order blocks for mean reversion in ranging markets.
In ranging conditions (price between weekly high/low), fades moves toward weekly liquidity zones with 1d order block confirmation.
Weekly trend filter avoids counter-trend trades during strong trends.
Works in both bull/bear markets as ranging occurs during consolidation periods across all cycles.
Target: 15-30 trades/year on 6h (60-120 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_order_block(high, low, close):
    """Calculate bullish and bearish order blocks from price action."""
    # Bullish OB: bearish candle followed by bullish candle that closes above midpoint
    bullish_ob = np.zeros_like(close, dtype=bool)
    bearish_ob = np.zeros_like(close, dtype=bool)
    
    for i in range(2, len(close)):
        # Bearish candle (current) followed by bullish candle (previous)
        if close[i-1] < open[i-1] and close[i] > open[i]:
            # Current candle is bullish, previous was bearish
            midpoint = (high[i-1] + low[i-1]) / 2
            if close[i] > midpoint:
                bullish_ob[i-1] = True  # Mark the bearish candle as bullish OB
        
        # Bullish candle (current) followed by bearish candle (previous)
        if close[i-1] > open[i-1] and close[i] < open[i]:
            # Current candle is bearish, previous was bullish
            midpoint = (high[i-1] + low[i-1]) / 2
            if close[i] < midpoint:
                bearish_ob[i-1] = True  # Mark the bullish candle as bearish OB
    
    return bullish_ob, bearish_ob

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_prices = prices['open'].values
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for liquidity zones (weekly high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    
    # Weekly trend: bullish if weekly close > weekly open
    weekly_bullish = weekly_close > weekly_open
    
    # Get daily data for order blocks
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate order blocks on daily
    bullish_ob_1d, bearish_ob_1d = calculate_order_block(high_1d, low_1d, close_1d)
    
    # Align all data to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    bullish_ob_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_1d.astype(float))
    bearish_ob_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_1d.astype(float))
    
    # Volume filter: current volume > 1.3x 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    volume_filter = volume > (vol_ma_24 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(bullish_ob_aligned[i]) or
            np.isnan(bearish_ob_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Ranging condition: price between weekly high and low
        in_range = (low[i] >= weekly_low_aligned[i]) and (high[i] <= weekly_high_aligned[i])
        
        if in_range:
            # Mean reversion at weekly liquidity zones with order block confirmation
            
            # Long setup: price near weekly low with bullish order block and weekly bullish bias
            near_weekly_low = low[i] <= (weekly_low_aligned[i] * 1.002)  # Within 0.2% of weekly low
            long_condition = near_weekly_low and bullish_ob_aligned[i] > 0.5 and weekly_bullish_aligned[i] > 0.5 and volume_filter[i]
            
            # Short setup: price near weekly high with bearish order block and weekly bearish bias
            near_weekly_high = high[i] >= (weekly_high_aligned[i] * 0.998)  # Within 0.2% of weekly high
            short_condition = near_weekly_high and bearish_ob_aligned[i] > 0.5 and weekly_bullish_aligned[i] < 0.5 and volume_filter[i]
            
            if long_condition and position != 1:
                position = 1
                signals[i] = position_size
            elif long_condition and position == 1:
                signals[i] = position_size
            elif short_condition and position != -1:
                position = -1
                signals[i] = -position_size
            elif short_condition and position == -1:
                signals[i] = -position_size
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Outside weekly range - exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_Liquidity_Capture"
timeframe = "6h"
leverage = 1.0
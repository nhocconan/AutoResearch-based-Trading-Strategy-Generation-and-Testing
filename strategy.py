#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# In ranging markets (common in 2025-2026), price tends to revert from R3/S3 levels.
# In trending markets, breakouts through R4/S4 with volume continue the trend.
# Uses 1d EMA50 to determine regime: above = bullish trend, below = bearish trend.
# Entry conditions: 
#   - Bullish: price crosses below S3 in bullish trend OR breaks above R4 with volume in any regime
#   - Bearish: price crosses above R3 in bearish trend OR breaks below S4 with volume in any regime
# Designed for low trade frequency (~15-25/year) with clear rules to avoid overtrading.
name = "6h_Camarilla_Reversal_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter (regime detection)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's OHLC for Camarilla calculation (using 1d data)
    prev_close = df_1d['close'].shift(1).values  # yesterday's close
    prev_high = df_1d['high'].shift(1).values    # yesterday's high
    prev_low = df_1d['low'].shift(1).values      # yesterday's low
    
    # Align previous day's OHLC to 6h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels for today (based on yesterday's price action)
    # Typical price = (high + low + close) / 3
    typical_price = (prev_high_aligned + prev_low_aligned + prev_close_aligned) / 3.0
    range_val = prev_high_aligned - prev_low_aligned
    
    # Camarilla levels
    R4 = close + (range_val * 1.500)  # Note: using current close as base, standard formula
    R3 = close + (range_val * 1.250)
    S3 = close - (range_val * 1.250)
    S4 = close - (range_val * 1.500)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(prev_close_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend regime: above EMA50 = bullish, below = bearish
        is_bullish_trend = close[i] > ema50_1d_aligned[i]
        
        if position == 0:
            # Long entry conditions:
            # 1. Reversal from S3 in bullish trend (price crosses above S3 from below)
            # 2. Breakout above R4 with volume (in any regime)
            reversal_long = (close[i] > S3[i] and close[i-1] <= S3[i-1] and is_bullish_trend)
            breakout_long = (close[i] > R4[i] and volume_filter[i])
            
            if reversal_long or breakout_long:
                signals[i] = 0.25
                position = 1
            # Short entry conditions:
            # 1. Reversal from R3 in bearish trend (price crosses below R3 from above)
            # 2. Breakout below S4 with volume (in any regime)
            elif (close[i] < R3[i] and close[i-1] >= R3[i-1] and not is_bullish_trend) or \
                 (close[i] < S4[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: reversal from R3 or breakdown below S3
            if (close[i] < R3[i] and close[i-1] >= R3[i-1]) or (close[i] < S3[i] and close[i-1] >= S3[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: reversal from S3 or breakout above R3
            if (close[i] > S3[i] and close[i-1] <= S3[i-1]) or (close[i] > R3[i] and close[i-1] <= R3[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
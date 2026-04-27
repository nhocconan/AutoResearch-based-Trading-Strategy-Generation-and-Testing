#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels on 1d provide institutional support/resistance. 
Breakout above R3 or below S3 with 1d EMA34 trend filter and volume surge captures 
strong moves. Works in bull (breakouts up) and bear (breakdowns down) by using 
1d trend direction. Target: 15-25 trades/year to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivots, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = calculate_ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d average volume for volume filter
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Need previous day's OHLC to calculate today's levels
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    # First value will be invalid (rolled), but alignment will handle timing
    
    # Camarilla calculations
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current 12h volume > 1.5x 1d average volume
    vol_confirm = volume > (vol_ma_1d_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA34 and rolled values
    start_idx = max(35, 20)  # EMA34 needs ~34, plus 1 for roll
    
    for i in range(start_idx, n):
        # Skip if any data not ready (first rolled values will be NaN)
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above 1d EMA34 (bullish trend), volume confirmation
            if close[i] > R3_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = size
                position = 1
            # Short: price breaks below S3, below 1d EMA34 (bearish trend), volume confirmation
            elif close[i] < S3_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 (reversal) or below EMA34 (trend change)
            if close[i] < S3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R3 (reversal) or above EMA34 (trend change)
            if close[i] > R3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0
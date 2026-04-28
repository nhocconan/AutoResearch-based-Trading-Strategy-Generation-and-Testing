#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_1wTrend
Hypothesis: On 12-hour timeframe, enter long when price touches Camarilla S1 level with bullish divergence (price > 1w EMA50) and volume spike; short when price touches R1 level with bearish divergence (price < 1w EMA50). Exit on opposite touch. Uses 1w EMA50 for trend filter and volume confirmation to reduce false signals. Targets 50-150 trades over 4 years to minimize fee drag while capturing reversals in ranging markets and continuations in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Focus on S1 and R1 for reversals
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all data to 12h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: bullish when price > 1w EMA50, bearish when price < 1w EMA50
        bullish_trend = close[i] > ema_50_1w_aligned[i]
        bearish_trend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: touch S1/R1 with trend alignment and volume spike
        long_entry = (low[i] <= S1_aligned[i]) and bullish_trend and volume_spike[i]
        short_entry = (high[i] >= R1_aligned[i]) and bearish_trend and volume_spike[i]
        
        # Exit conditions: touch opposite level
        long_exit = high[i] >= R1_aligned[i]
        short_exit = low[i] <= S1_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_1wTrend"
timeframe = "12h"
leverage = 1.0
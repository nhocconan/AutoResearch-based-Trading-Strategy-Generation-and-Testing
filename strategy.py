#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Camarilla pivot levels on daily timeframe identify key support/resistance.
Breakouts above resistance or below support with volume expansion signal strong momentum.
Weekly timeframe filters for trend direction (price above/below weekly EMA20).
Works in bull markets (breakouts above resistance) and bear markets (breakdowns below support).
Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas: 
    # Resistance 4 = C + ((H-L) * 1.1/2)
    # Resistance 3 = C + ((H-L) * 1.1/4)
    # Resistance 2 = C + ((H-L) * 1.1/6)
    # Resistance 1 = C + ((H-L) * 1.1/12)
    # Support 1 = C - ((H-L) * 1.1/12)
    # Support 2 = C - ((H-L) * 1.1/6)
    # Support 3 = C - ((H-L) * 1.1/4)
    # Support 4 = C - ((H-L) * 1.1/2)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate levels
    r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    r2 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    s2 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align to lower timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above R1 (first resistance level)
        # 2. Price above weekly EMA20 (weekly trend filter)
        # 3. Volume expansion
        breakout_long = close[i] > r1_aligned[i]
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        long_condition = breakout_long and price_above_weekly_ema and volume_expansion[i]
        
        # Short conditions:
        # 1. Price breaks below S1 (first support level)
        # 2. Price below weekly EMA20 (weekly trend filter)
        # 3. Volume expansion
        breakdown_short = close[i] < s1_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        short_condition = breakdown_short and price_below_weekly_ema and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0
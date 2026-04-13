# 4h_1d_Camarilla_Breakout_Structure_v1
# Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
# Breakouts above R3 or below S3 with volume confirmation and aligned with weekly trend
# (price > weekly EMA200 for longs, price < weekly EMA200 for shorts) capture institutional moves.
# Works in bull markets (breakouts continue) and bear markets (fades at resistance) by using
# price action confirmation rather than pure breakout logic. Target: 20-40 trades/year.

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar: H-L range based
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    r4_1d = close_1d + 1.5 * range_1d
    s4_1d = close_1d - 1.5 * range_1d
    
    # Align 1d Camarilla levels to 4h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Breakout above R3 with volume expansion
        # 2. Must be above weekly EMA200 for trend alignment
        breakout_long = (close[i] > r3_1d_aligned[i]) and volume_expansion[i]
        trend_long = close[i] > ema_200_aligned[i]
        long_condition = breakout_long and trend_long
        
        # Short conditions:
        # 1. Breakdown below S3 with volume expansion
        # 2. Must be below weekly EMA200 for trend alignment
        breakdown_short = (close[i] < s3_1d_aligned[i]) and volume_expansion[i]
        trend_short = close[i] < ema_200_aligned[i]
        short_condition = breakdown_short and trend_short
        
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

name = "4h_1d_Camarilla_Breakout_Structure_v1"
timeframe = "4h"
leverage = 1.0
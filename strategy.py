#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Price_Action_Pivot_Reversal_1dTrend_Volume"
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
    
    # Get 1d data once for trend filter and pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate daily pivots and support/resistance levels
    # Pivot Point = (High + Low + Close) / 3
    # R1 = 2*P - Low, S1 = 2*P - High
    # R2 = P + (High - Low), S2 = P - (High - Low)
    # R3 = High + 2*(P - Low), S3 = Low - 2*(High - P)
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pp - df_1d['low']
    s1 = 2 * pp - df_1d['high']
    r2 = pp + (df_1d['high'] - df_1d['low'])
    s2 = pp - (df_1d['high'] - df_1d['low'])
    r3 = df_1d['high'] + 2 * (pp - df_1d['low'])
    s3 = df_1d['low'] - 2 * (df_1d['high'] - pp)
    
    # Align pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price near S1/S2/S3 with volume spike and daily uptrend
            near_support = (low[i] <= s1_aligned[i] * 1.005 or 
                           low[i] <= s2_aligned[i] * 1.005 or 
                           low[i] <= s3_aligned[i] * 1.005)
            
            # Short setup: price near R1/R2/R3 with volume spike and daily downtrend
            near_resistance = (high[i] >= r1_aligned[i] * 0.995 or 
                              high[i] >= r2_aligned[i] * 0.995 or 
                              high[i] >= r3_aligned[i] * 0.995)
            
            if near_support and vol_spike[i] and trend_1d_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            elif near_resistance and vol_spike[i] and trend_1d_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches R1 or daily trend turns down
            if (high[i] >= r1_aligned[i] * 0.995 or trend_1d_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S1 or daily trend turns up
            if (low[i] <= s1_aligned[i] * 1.005 or trend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Price action reversal at daily pivot points (S1/S2/S3 for longs, R1/R2/R3 for shorts) 
# with volume confirmation and daily trend filter on 6h timeframe. 
# Works in ranging markets (reversions at pivot levels) and trending markets 
# (breakouts when price breaks through pivot levels with volume). 
# Daily EMA50 ensures alignment with medium-term trend, reducing counter-trend trades. 
# Volume spike requirement (2x average) ensures institutional participation. 
# Target: 20-40 trades/year to minimize fee decay while capturing meaningful moves. 
# Pivot levels provide objective support/resistance that work across market regimes. 
# The strategy avoids chop by requiring both proximity to pivot levels AND volume spike. 
# Exit conditions are based on reaching opposite pivot level or trend change.
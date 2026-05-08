#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend direction
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 4h daily pivot points for S1/R1 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Align daily pivot to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema20_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above S1 with volume spike, above weekly EMA
            if (close[i] > s1_aligned[i] and vol_spike and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below R1 with volume spike, below weekly EMA
            elif (close[i] < r1_aligned[i] and vol_spike and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR below weekly EMA
            if (close[i] < s1_aligned[i] or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 OR above weekly EMA
            if (close[i] > r1_aligned[i] or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla S1/R1 breakout with volume confirmation and 1w EMA trend filter.
# - Enters long when price breaks above S1 (daily pivot support) with volume spike and above weekly EMA
# - Enters short when price breaks below R1 (daily pivot resistance) with volume spike and below weekly EMA
# - Uses weekly trend filter to ensure alignment with higher timeframe direction
# - Volume spike requirement reduces false breakouts in low-volume environments
# - Daily pivot points provide reliable support/resistance levels
# - Position size 0.25 balances risk and return while minimizing fee churn
# - Target: 20-50 trades per year to stay within optimal frequency range
# - Works in both bull and bear markets by following weekly trend direction
# - Simple 3-condition logic avoids overfitting and excessive trading
# - Weekly EMA provides smooth trend filter that adapts to changing market conditions
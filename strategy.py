#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d/1w trend filter
# - Uses 6h Williams %R(14) for oversold/overbought signals (long < -80, short > -20)
# - Filters with 1d EMA(50) trend: only long when price > EMA50, short when price < EMA50
# - Confirms with 1w Williams %R(14) extreme: avoid counter-trend trades in strong weekly momentum
# - Exits when Williams %R reverts to mean (-50) or opposite extreme
# - Position size: 0.25 (25% of capital) to limit drawdown in volatile 6h markets
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years) to minimize fee drag
# - Williams %R captures short-term reversals while higher timeframe filters prevent
#   trading against the dominant trend, working in both bull and bear markets

name = "6h_1d_1w_williamsr_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Williams %R(14)
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w + 1e-10)
    
    # Align 1d and 1w indicators to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    williams_r_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(williams_r_1w_aligned[i]) or ema_50_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R reverts to mean (-50) or reaches overbought (-20)
            if williams_r[i] >= -50:  # Mean reversion exit
                position = 0
                signals[i] = 0.0
            elif williams_r[i] >= -20:  # Overbought exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R reverts to mean (-50) or reaches oversold (-80)
            if williams_r[i] <= -50:  # Mean reversion exit
                position = 0
                signals[i] = 0.0
            elif williams_r[i] <= -80:  # Oversold exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with trend filter
            # Long: oversold (-80) + price above 1d EMA50 + not in strong weekly uptrend
            if (williams_r[i] <= -80 and  # Oversold
                close[i] > ema_50_1d_aligned[i] and  # Above 1d EMA50 (uptrend filter)
                williams_r_1w_aligned[i] > -80):  # Not in strong weekly uptrend (> -80)
                position = 1
                signals[i] = 0.25
            # Short: overbought (-20) + price below 1d EMA50 + not in strong weekly downtrend
            elif (williams_r[i] >= -20 and  # Overbought
                  close[i] < ema_50_1d_aligned[i] and  # Below 1d EMA50 (downtrend filter)
                  williams_r_1w_aligned[i] < -20):  # Not in strong weekly downtrend (< -20)
                position = -1
                signals[i] = -0.25
    
    return signals
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w trend filter for mean reversion in higher timeframes
# - Williams %R(14) on daily: long when <-80 (oversold), short when >-20 (overbought)
# - 1w EMA(50) trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50
# - This avoids counter-trend trades in strong trends, improving win rate
# - Designed for 1d timeframe to reduce trade frequency and minimize fee drag
# - Target: 15-25 trades/year to stay within optimal range for 1d

name = "1d_WilliamsR_1wTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R(14) on daily data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Look for long entry: uptrend (price > 1w EMA50) + oversold Williams %R
            if close[i] > ema_50_1w_aligned[i] and williams_r[i] < -80:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend (price < 1w EMA50) + overbought Williams %R
            elif close[i] < ema_50_1w_aligned[i] and williams_r[i] > -20:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on overbought Williams %R or trend reversal
            if williams_r[i] > -20 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on oversold Williams %R or trend reversal
            if williams_r[i] < -80 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
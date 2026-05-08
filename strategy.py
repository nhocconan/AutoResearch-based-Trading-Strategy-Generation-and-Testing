#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d Trend + Volume Spike
# - Williams %R(14) identifies overbought/oversold conditions
# - In strong uptrend (1d EMA50 > 1d EMA200), buy when %R crosses above -80 from below
# - In strong downtrend (1d EMA50 < 1d EMA200), sell when %R crosses below -20 from above
# - Volume spike (>2x 20-period average) confirms momentum
# - Works in bull/bear by following 1d trend, avoiding counter-trend trades
# - Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag on 4h timeframe

name = "4h_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Williams %R(14) calculation
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 1 = uptrend, -1 = downtrend
        if ema_50_1d_aligned[i] > ema_200_1d_aligned[i]:
            trend = 1
        else:
            trend = -1
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below in uptrend with volume spike
            long_cond = (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                        trend == 1 and volume_spike[i])
            
            # Short: Williams %R crosses below -20 from above in downtrend with volume spike
            short_cond = (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                         trend == -1 and volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 or trend reverses
            if williams_r[i] < -50 or ema_50_1d_aligned[i] < ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 or trend reverses
            if williams_r[i] > -50 or ema_50_1d_aligned[i] > ema_200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
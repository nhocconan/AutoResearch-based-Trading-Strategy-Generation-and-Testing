#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 from below in 1d uptrend (price > EMA50).
# Short when Williams %R crosses below -20 from above in 1d downtrend (price < EMA50).
# Volume must be > 1.8x 20-period MA to confirm reversal strength.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 total trades over 4 years.
# Williams %R is an oscillator that identifies overbought/oversold conditions.
# In strong trends, it can remain extreme for extended periods, but reversals from
# extreme levels often signal profitable entry points with good risk-reward.
# The 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend,
# avoiding counter-trend trades during both bull and bear markets.
# Volume confirmation ensures the reversal has sufficient participation.

name = "6h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        williams_r_val = williams_r[i]
        trend_up = close_val > ema_50_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_50_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        
        # Williams %R reversal signals
        williams_r_oversold = williams_r_val <= -80
        williams_r_overbought = williams_r_val >= -20
        
        # Entry logic
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND 1d uptrend AND volume spike
            if (williams_r_val > -80 and williams_r[i-1] <= -80 and 
                trend_up and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND 1d downtrend AND volume spike
            elif (williams_r_val < -20 and williams_r[i-1] >= -20 and 
                  trend_down and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 OR 1d trend turns down
            if (williams_r_val >= -20 and williams_r[i-1] < -20) or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 OR 1d trend turns up
            if (williams_r_val <= -80 and williams_r[i-1] > -80) or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
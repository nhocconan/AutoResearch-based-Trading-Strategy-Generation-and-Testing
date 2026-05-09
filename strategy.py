#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with weekly trend filter and volume confirmation
# Uses Williams %R(14) from 1d for overbought/oversold signals, weekly EMA34 for trend alignment,
# and volume spike for confirmation. Works in bull markets (oversold bounces in uptrend) and bear markets
# (overbought reversals in downtrend). Designed for 15-35 trades/year to avoid fee drag.
name = "6h_WilliamsR_WeeklyTrend_Volume"
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
    
    # Get daily data for Williams %R calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    ema34_weekly = pd.Series(df_weekly['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Williams %R(14) calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_daily['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_daily['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_daily['close'].values) / (highest_high - lowest_low)
    williams_r_6h = align_htf_to_ltf(prices, df_daily, williams_r)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_6h[i]) or np.isnan(williams_r_6h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with weekly uptrend and volume spike
            if williams_r_6h[i] < -80 and close[i] > ema34_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with weekly downtrend and volume spike
            elif williams_r_6h[i] > -20 and close[i] < ema34_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 OR weekly trend turns down
            if williams_r_6h[i] > -50 or close[i] < ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 OR weekly trend turns up
            if williams_r_6h[i] < -50 or close[i] > ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
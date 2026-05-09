#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h momentum strategy using 1-day Williams %R for oversold/overbought conditions
# with volume confirmation and 1-day EMA trend filter. Designed to capture mean reversion
# within the trend on 12h timeframe. Target: 15-35 trades/year to avoid fee drag.
# Works in bull (buy oversold in uptrend) and bear (sell overbought in downtrend).
name = "12h_WilliamsR_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Williams %R and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    
    # 1-day EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Williams %R and EMA to 12h timeframe (use previous day's values)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + price above EMA (uptrend) + volume confirmation
            if (williams_r_aligned[i] < -80 and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + price below EMA (downtrend) + volume confirmation
            elif (williams_r_aligned[i] > -20 and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 (momentum fading) or trend reverses
            if williams_r_aligned[i] > -50 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 (momentum fading) or trend reverses
            if williams_r_aligned[i] < -50 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
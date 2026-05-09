#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour chart Williams %R overbought/oversold signals with 1-day EMA trend filter and volume confirmation.
# Williams %R identifies momentum extremes; EMA filter ensures alignment with daily trend; volume confirms strength.
# This combination works in both bull and bear markets by following the higher-timeframe trend while capturing mean reversion within that trend.
name = "12h_WilliamsR_1dEMA34_Volume"
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
    
    # Get 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough for EMA(34)
        return np.zeros(n)
    
    # Calculate Williams %R on 12-hour data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We'll use 14-period lookback as standard
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Calculate 1-day EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # Need enough data for Williams %R calculation
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from calculations or alignment)
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above 1-day EMA34 + volume confirmation
            if (williams_r[i] < -80 and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price below 1-day EMA34 + volume confirmation
            elif (williams_r[i] > -20 and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 (momentum fading) or price breaks below EMA
            if (williams_r[i] > -50 or price < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 (momentum fading) or price breaks above EMA
            if (williams_r[i] < -50 or price > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
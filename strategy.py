#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams %R extreme reversal with 1-day trend filter and volume confirmation.
Enter long when Williams %R crosses above -80 from oversold during 1-day uptrend with volume spike.
Enter short when Williams %R crosses below -20 from overbought during 1-day downtrend with volume spike.
Williams %R identifies overextended moves likely to reverse, while the daily trend filter ensures
we trade in the direction of higher timeframe momentum. Volume confirmation avoids false reversals.
Designed for moderate trade frequency (20-50 trades/year) by requiring confluence of momentum,
trend, and volume. Works in both bull and bear markets by following the 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 4h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Daily EMA50 for trend direction
    daily_close = df_daily['close'].values
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema50_daily_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (exiting oversold) + daily uptrend + volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                ema50_daily_aligned[i] > ema50_daily_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (exiting overbought) + daily downtrend + volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  ema50_daily_aligned[i] < ema50_daily_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to opposite extreme or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -20 (overbought) or daily trend turns down
                if williams_r[i] >= -20 or ema50_daily_aligned[i] < ema50_daily_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -80 (oversold) or daily trend turns up
                if williams_r[i] <= -80 or ema50_daily_aligned[i] > ema50_daily_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0
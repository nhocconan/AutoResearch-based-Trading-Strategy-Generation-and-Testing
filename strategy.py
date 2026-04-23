#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA200 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND daily close > daily EMA200 AND volume > 2x average.
Short when Williams %R > -20 (overbought) AND daily close < daily EMA200 AND volume > 2x average.
Exit when Williams %R crosses above -50 (for long) or below -50 (for short).
Williams %R identifies extreme reversals, daily EMA200 filter ensures trend alignment,
volume spike confirms institutional interest. Targets 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA200 on 1d data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Williams %R parameters
    williams_period = 14
    
    # Calculate Williams %R on 6h data
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        daily_trend_up = close[i] > ema200_1d_aligned[i]
        daily_trend_down = close[i] < ema200_1d_aligned[i]
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND daily uptrend AND volume spike
            if (williams_r[i] < -80 and daily_trend_up and 
                vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND daily downtrend AND volume spike
            elif (williams_r[i] > -20 and daily_trend_down and 
                  vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50 (mean reversion midpoint)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50
                if williams_r[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1dEMA200_VolumeSpike"
timeframe = "6h"
leverage = 1.0
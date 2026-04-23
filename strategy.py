#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1w trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND weekly close > weekly EMA34 AND volume > 2.0x average.
Short when Williams %R > -20 (overbought) AND weekly close < weekly EMA34 AND volume > 2.0x average.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year per symbol.
Williams %R captures momentum exhaustion, weekly trend filter ensures directional alignment,
volume spike confirms institutional participation at extremes.
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
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams %R (14-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_trend_up = close[i] > ema34_1w_aligned[i]
        weekly_trend_down = close[i] < ema34_1w_aligned[i]
        
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND weekly uptrend AND volume spike
            if (williams_r[i] < -80 and weekly_trend_up and 
                vol_current > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND weekly downtrend AND volume spike
            elif (williams_r[i] > -20 and weekly_trend_down and 
                  vol_current > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (momentum fading)
                if williams_r[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (momentum fading)
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1wEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0
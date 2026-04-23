#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d EMA200 trend filter and volume spike confirmation.
In strong trends (price > 1d EMA200 for long, price < 1d EMA200 for short), enter when Williams %R indicates oversold/overbought conditions (below -80 for long, above -20 for short) with volume > 1.5x 20-period average.
Exit when Williams %R returns to neutral range (-50 to -50) or trend reverses.
Uses discrete position sizing (0.25) to limit fee drag and targets 12-30 trades/year on 12h timeframe.
Designed to work in both bull and bear markets by trading mean reversion within the dominant trend.
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
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate 12h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 12h volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 14, 20)  # need EMA200, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: oversold (Williams %R < -80) AND uptrend (price > 1d EMA200) AND volume spike
            if williams_r[i] < -80 and close[i] > ema_200_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought (Williams %R > -20) AND downtrend (price < 1d EMA200) AND volume spike
            elif williams_r[i] > -20 and close[i] < ema_200_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50 to -50) OR trend reversal
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R >= -50 (reversion) OR trend breaks (price <= 1d EMA200)
                if williams_r[i] >= -50 or close[i] <= ema_200_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R <= -50 (reversion) OR trend breaks (price >= 1d EMA200)
                if williams_r[i] <= -50 or close[i] >= ema_200_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_MeanReversion_1dEMA200_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0
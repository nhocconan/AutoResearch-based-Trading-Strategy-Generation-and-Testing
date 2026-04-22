#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R extreme reversals with 12-hour trend filter and volume confirmation.
Long when Williams %R crosses above -80 from oversold with 12-hour EMA50 rising and volume spike.
Short when Williams %R crosses below -20 from overbought with 12-hour EMA50 falling and volume spike.
Exit when Williams %R returns to -50 (mean reversion center).
Williams %R identifies overextended moves; 12-hour EMA50 filters for higher-timeframe trend;
volume spike confirms momentum exhaustion. Designed for low frequency by requiring overextension
plus trend alignment plus volume. Works in both bull (buy oversold dips) and bear (sell overbought rallies).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for Williams %R and EMA50 - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    period = 14
    highest_high = pd.Series(high_12h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low)
    
    # 12-hour EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from oversold with 12h EMA50 rising and volume spike
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought with 12h EMA50 falling and volume spike
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to -50 (mean reversion center)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50
                if williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -50
                if williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extreme_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0
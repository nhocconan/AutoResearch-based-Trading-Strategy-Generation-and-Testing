#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d EMA200 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA200 AND volume > 1.3x average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA200 AND volume > 1.3x average.
Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts).
Williams %R identifies extreme price levels, 1d EMA200 ensures higher timeframe trend alignment,
volume confirmation filters low-conviction moves. Designed for 4h targeting 75-200 total trades.
Works in both bull and bear markets by fading extremes in direction of 1d trend.
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
    
    # Load 1d data for EMA200 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d data
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate Williams %R on 4h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200_val = ema200_1d_aligned[i]
        wr = williams_r[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA200 AND volume spike
            if (wr < -80 and close[i] > ema200_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA200 AND volume spike
            elif (wr > -20 and close[i] < ema200_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses back above -50
                if wr > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses back below -50
                if wr < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0
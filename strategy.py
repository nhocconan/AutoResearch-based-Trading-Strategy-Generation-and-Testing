#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 12h EMA20 trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) AND price > 12h EMA20 AND volume > 1.3x average.
Short when Williams %R > -20 (overbought) AND price < 12h EMA20 AND volume > 1.3x average.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Williams %R identifies exhaustion points; 12h EMA20 ensures trading with higher timeframe trend.
Volume confirmation filters low-conviction moves. Designed for 6h timeframe targeting 50-150 total trades over 4 years.
Works in both bull and bear markets by taking mean-reversion trades aligned with 12h trend.
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
    
    # Load 12h data for EMA20 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA20 on 12h data
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA20 to 6h timeframe
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_val = ema20_12h_aligned[i]
        wr_val = williams_r[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > 12h EMA20 AND volume spike
            if (wr_val < -80 and price > ema20_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 12h EMA20 AND volume spike
            elif (wr_val > -20 and price < ema20_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (momentum fading)
                if wr_val > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (momentum fading)
                if wr_val < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_12hEMA20_Volume"
timeframe = "6h"
leverage = 1.0
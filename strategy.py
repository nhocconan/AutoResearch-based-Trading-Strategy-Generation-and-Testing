#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R with 14-period and 1d EMA 50 for trend filter.
# Long when 12h Williams %R < -80 (oversold) and price above 1d EMA50 with volume confirmation.
# Short when 12h Williams %R > -20 (overbought) and price below 1d EMA50 with volume confirmation.
# Exit when Williams %R crosses back to -50.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Works in both trending and ranging markets via trend filter.

name = "6h_12hWilliamsR_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    highest_high = np.zeros_like(high_12h)
    lowest_low = np.zeros_like(low_12h)
    
    for i in range(len(high_12h)):
        if i < 13:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high_12h[i-13:i+1])
            lowest_low[i] = np.min(low_12h[i-13:i+1])
    
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close_12h) / (highest_high - lowest_low), 
                          -50)
    
    # Get 1d data for EMA 50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = np.zeros_like(close_1d)
    
    # Calculate EMA 50
    ema_50[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_50[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50[i-1] * (48 / (50 + 1)))
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 6h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 and price above 1d EMA50 with volume confirmation
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 and price below 1d EMA50 with volume confirmation
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
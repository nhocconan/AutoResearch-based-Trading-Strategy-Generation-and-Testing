#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R with 14-period for mean reversion in range-bound markets.
# Uses 1d Williams %R to detect overbought/oversold conditions with volume confirmation and 1d EMA50 trend filter.
# Long when Williams %R < -80 and price above EMA50 with volume confirmation.
# Short when Williams %R > -20 and price below EMA50 with volume confirmation.
# Exit when Williams %R crosses back to -50 level.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Works in both trending and ranging markets via trend filter.

name = "4h_1dWilliamsR_EMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    
    # For Williams %R, we need the highest high and lowest low over the last 14 periods
    williams_r = np.full_like(close_1d, -50.0, dtype=np.float64)
    
    for i in range(13, len(close_1d)):
        period_high = np.max(high_1d[i-13:i+1])
        period_low = np.min(low_1d[i-13:i+1])
        if period_high != period_low:
            williams_r[i] = -100 * (period_high - close_1d[i]) / (period_high - period_low)
        else:
            williams_r[i] = -50.0
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 4h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for Williams %R and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R < -80 and price above EMA50 with volume confirmation
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R > -20 and price below EMA50 with volume confirmation
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
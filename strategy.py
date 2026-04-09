#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend filter + volume confirmation
# Uses Williams %R(14) on 6h for overbought/oversold signals, confirmed by 1d EMA(50) trend direction
# Only takes trades when 6h volume > 1.5x 20-period average for conviction
# In bull markets (price > 1d EMA): long when WR < -80, exit at WR > -20
# In bear markets (price < 1d EMA): short when WR > -20, exit at WR < -80
# Position size 0.25 to manage drawdown and enable multiple concurrent positions
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: 1d EMA filter ensures we trade with the higher timeframe trend

name = "6h_1d_williamsr_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d_50 = np.full(len(df_1d), np.nan)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_1d_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d_50[i] = (close_1d[i] * multiplier) + (ema_1d_50[i-1] * (1 - multiplier))
    
    # Calculate 6h Williams %R(14)
    williams_r = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            williams_r[i] = np.nan
        else:
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high - lowest_low != 0:
                williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                williams_r[i] = np.nan
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d EMA to 6h timeframe
    ema_1d_50_6h = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_1d_50_6h[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        # Trend filter: price relative to 1d EMA(50)
        trend_up = close[i] > ema_1d_50_6h[i]
        trend_down = close[i] < ema_1d_50_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions: Williams %R > -20 (overbought) OR volume confirmation fails
            if williams_r[i] > -20 or not volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Williams %R < -80 (oversold) OR volume confirmation fails
            if williams_r[i] < -80 or not volume_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extreme with volume confirmation and trend filter
            if volume_confirm:
                # Long entry: oversold in uptrend
                if williams_r[i] < -80 and trend_up:
                    position = 1
                    signals[i] = 0.25
                # Short entry: overbought in downtrend
                elif williams_r[i] > -20 and trend_down:
                    position = -1
                    signals[i] = -0.25
    
    return signals